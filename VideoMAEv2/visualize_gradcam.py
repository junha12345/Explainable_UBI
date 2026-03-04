
import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from timm.models import create_model

# Add current directory to path
sys.path.append(os.getcwd())

import models
from dataset import video_transforms, volume_transforms
import matplotlib.cm as cm

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_forward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return
        
        def hook_function(grad):
            self.gradients = grad
            
        output.register_hook(hook_function)

    def __call__(self, x, class_idx=None):
        # Forward pass
        self.model.zero_grad()
        logits = self.model(x)
        
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        
        # Target score
        one_hot = torch.zeros_like(logits)
        one_hot[0][class_idx] = 1
        
        # Backward
        logits.backward(gradient=one_hot, retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        
        # Gradients: (B, N, C)
        # Activations: (B, N, C)
        
        # Global Average Pooling of gradients across spatial/temporal dimensions (N)
        # weights: (B, C)
        weights = torch.mean(gradients, dim=1, keepdim=True)
        
        # Weighted combination of activations
        # (B, N, C) * (B, 1, C) -> (B, N, C) -> sum dim 2 -> (B, N)
        cam = torch.sum(activations * weights, dim=2)
        
        # ReLU
        cam = F.relu(cam)
        
        # Restore shape
        # N = T_feat * H_feat * W_feat + (potentially CLS token)
        # If CLS token exists (N=1569 or similar), ignore it for spatial visualization
        return cam, class_idx

def get_args():
    parser = argparse.ArgumentParser('VideoMAE Grad-CAM')
    parser.add_argument('--checkpoint', default='work_dirs/custom_finetune/checkpoint-19.pth', type=str)
    parser.add_argument('--video_path', required=True, type=str)
    parser.add_argument('--output_dir', default='test_video_vis', type=str)
    parser.add_argument('--model', default='vit_base_patch16_224', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--num_frames', default=16, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--target_class', default=None, type=int, help='Force a target class index')
    return parser.parse_args()

def load_video_decord(sample, num_frames=16):
    from decord import VideoReader, cpu
    vr = VideoReader(sample, ctx=cpu(0))
    length = len(vr)
    all_index = np.linspace(0, length-1, num_frames).astype(int)
    vr.seek(0)
    buffer = vr.get_batch(all_index).asnumpy()
    return buffer, all_index

def reshape_transform(tensor, t_feat, h_feat, w_feat):
    # tensor: (B, N)
    # Remove CLS token if present
    # expected N = t*h*w
    
    B, N = tensor.shape
    expected = t_feat * h_feat * w_feat
    
    if N == expected:
        pass
    elif N == expected + 1:
        tensor = tensor[:, 1:]
    else:
        # Fallback? attempt to slice
        tensor = tensor[:, 1:]
        
    result = tensor.reshape(B, t_feat, h_feat, w_feat)
    return result

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
        all_frames=args.num_frames,
        tubelet_size=2,
    )
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        print("Model loaded.")
    else:
        print("Checkpoint not found.")
        return

    model.to(device)
    model.eval()
    
    # Target Layer: Last Block
    # model.blocks[-1]
    # In ViT, blocks is a ModuleList
    target_layer = model.blocks[-1]
    grad_cam = GradCAM(model, target_layer)
    
    # Load Video
    frames, _ = load_video_decord(args.video_path, num_frames=args.num_frames)
    
    # Transform
    transform_pipeline = video_transforms.Compose([
        video_transforms.Resize(256, interpolation='bilinear'),
        video_transforms.CenterCrop(size=(args.input_size, args.input_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    pil_frames = [Image.fromarray(f) for f in frames]
    data = transform_pipeline(pil_frames)
    data = data.unsqueeze(0).to(device)
    data.requires_grad = True
    
    # Run Grad-CAM
    cam, pred_class = grad_cam(data, class_idx=args.target_class)
    
    val = pred_class.item() if isinstance(pred_class, torch.Tensor) else pred_class
    print(f"Prediction Class: {val} (Target Class: {args.target_class})")
    
    # Reshape CAM
    t_feat = args.num_frames // 2
    h_feat = args.input_size // 16
    w_feat = args.input_size // 16
    
    cam = reshape_transform(cam, t_feat, h_feat, w_feat) # (B, T, H, W)
    cam = cam[0].cpu().detach().numpy() # (T, H, W)
    
    # Normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    # Visualization
    os.makedirs(args.output_dir, exist_ok=True)
    video_name = os.path.basename(args.video_path)
    output_path = os.path.join(args.output_dir, f"gradcam_{video_name}")
    
    crop_transform = video_transforms.Compose([
        video_transforms.Resize(256),
        video_transforms.CenterCrop(args.input_size)
    ])
    
    vis_frames = []
    
    for i in range(args.num_frames):
        orig_img = pil_frames[i]
        orig_img = crop_transform([orig_img])[0]
        orig_np = np.array(orig_img)
        
        # Map frame to cam feature map
        cam_idx = i // 2
        cam_map = cam[cam_idx]
        
        cam_map_resized = cv2.resize(cam_map, (args.input_size, args.input_size), interpolation=cv2.INTER_CUBIC)
        
        heatmap = cm.jet(cam_map_resized)[..., :3] * 255.0
        heatmap = heatmap.astype(np.uint8)
        
        alpha = 0.5
        overlay = cv2.addWeighted(orig_np, 1-alpha, heatmap, alpha, 0)
        vis_frames.append(overlay)
        
    height, width, layers = vis_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 5.0, (width, height))
    
    for frame in vis_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Saved Grad-CAM to {output_path}")

if __name__ == '__main__':
    main()
