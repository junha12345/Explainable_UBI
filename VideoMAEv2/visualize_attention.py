
import os
import sys
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Add current directory to path so we can import models and dataset
sys.path.append(os.getcwd())

import models
from timm.models import create_model
from dataset import video_transforms, volume_transforms

def get_args():
    parser = argparse.ArgumentParser('VideoMAE Attention Visualization')
    parser.add_argument('--checkpoint', default='work_dirs/custom_finetune/checkpoint-19.pth', type=str)
    parser.add_argument('--video_path', required=True, type=str)
    parser.add_argument('--output_dir', default='test_video', type=str)
    parser.add_argument('--model', default='vit_base_patch16_224', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--num_frames', default=16, type=int)
    parser.add_argument('--sampling_rate', default=4, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    return parser.parse_args()

def load_video_decord(sample, sample_rate_scale=1, num_frames=16):
    from decord import VideoReader, cpu
    vr = VideoReader(sample, ctx=cpu(0))
    length = len(vr)
    # Temporal sampling (center clip or uniform)
    # We'll use uniform sampling for visualization to cover the video
    # But recall the model was trained with specific sampling. 
    # Let's try to match the training sampling roughly or just take 16 evenly spaced frames.
    
    # Simple uniform sampling
    all_index = np.linspace(0, length-1, num_frames).astype(int)
    
    vr.seek(0)
    buffer = vr.get_batch(all_index).asnumpy()
    return buffer, all_index

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes, 
        all_frames=args.num_frames,
        tubelet_size=2,
    )
    
    # Load Checkpoint
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        
        # Handle state dict (remove 'module.' prefix if present)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # Clean state dict keys
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
            
        # We need to handle potential shape mismatches if the head was different
        # but for loading base model it should be okay.
        # Strict=False to ignore head mismatch if classes differ
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"Load message: {msg}")
    else:
        print(f"Checkpoint not found at {args.checkpoint}")
        return

    model.to(device)
    model.eval()

    # 2. Load and Preprocess Video
    print(f"Processing video: {args.video_path}")
    frames, frame_indices = load_video_decord(args.video_path, num_frames=args.num_frames)
    
    # Prepare transforms
    # Resize -> CenterCrop -> Normalize
    # We do manual transform to keep copies of original frames for visualization
    
    transform_pipeline = video_transforms.Compose([
        video_transforms.Resize(256, interpolation='bilinear'),
        video_transforms.CenterCrop(size=(args.input_size, args.input_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # To PIL for transform
    pil_frames = [Image.fromarray(f) for f in frames]
    
    # Apply transform
    # Note: transform_pipeline expects list of PIL images
    data = transform_pipeline(pil_frames) # (C, T, H, W)
    data = data.unsqueeze(0).to(device) # (1, C, T, H, W)

    # 3. Forward Pass
    with torch.no_grad():
        _ = model(data)
        
        # 4. Extract Attention
        # Get last block's attention
        # Shape: (B, num_heads, N, N) where N = 1 + (T/2 * H/16 * W/16)
        # Using the last_attn we patched in
        attn = model.blocks[-1].attn.last_attn
        
    print(f"Attention shape: {attn.shape}")
    
    # Average across heads
    attn = attn.mean(dim=1) # (B, N, N)
    
    # Check for CLS token
    # If N == t_feat * h_feat * w_feat, then no CLS token
    # If N == 1 + ..., then CLS token present
    
    t_feat = args.num_frames // 2 # tubelet size 2
    h_feat = args.input_size // 16 # patch size 16
    w_feat = args.input_size // 16
    
    expected_patches = t_feat * h_feat * w_feat
    actual_tokens = attn.shape[-1]
    
    if actual_tokens == expected_patches:
        print("No CLS token detected. Using average attention map.")
        # Compute the average attention received by each token from all other tokens
        # A_ij is attention from i to j. We want 'j' importance.
        # Sum over i (sources).
        cls_attn = attn[0].mean(dim=0) # (N,)
    elif actual_tokens == expected_patches + 1:
        print("CLS token detected.")
        cls_attn = attn[0, 0, 1:] 
    else:
        raise RuntimeError(f"Mismatch in token count: expected {expected_patches} or {expected_patches+1}, got {actual_tokens}")
    
    cls_attn = cls_attn.reshape(t_feat, h_feat, w_feat)
    
    # Normalize attention mask for visualization
    cls_attn = cls_attn.cpu().numpy()
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())
    
    # 5. Visualization
    # We need to map attention back to T frames.
    # Since tubelet size is 2, each attention map corresponds to 2 frames.
    # We can duplicate it.
    
    os.makedirs(args.output_dir, exist_ok=True)
    video_name = os.path.basename(args.video_path)
    output_path = os.path.join(args.output_dir, f"vis_{video_name}")
    
    # Original frames for background
    # We need to apply the same Resize/CenterCrop to original frames to match
    crop_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.input_size)
    ])
    
    vis_frames = []
    
    for i in range(args.num_frames):
        # Get original frame corresponding to this index
        orig_img = pil_frames[i]
        orig_img = crop_transform(orig_img)
        orig_np = np.array(orig_img)
        
        # Get attention map
        # Map frame index i to attention index t
        attn_idx = i // 2
        attn_map = cls_attn[attn_idx] # (14, 14)
        
        # Upscale attention map to image size
        attn_map_resized = cv2.resize(attn_map, (args.input_size, args.input_size), interpolation=cv2.INTER_CUBIC)
        
        # Apply colormap
        heatmap = cm.jet(attn_map_resized)[..., :3] * 255.0
        heatmap = heatmap.astype(np.uint8)
        
        # Overlay
        # alpha blending
        alpha = 0.5
        overlay = cv2.addWeighted(orig_np, 1-alpha, heatmap, alpha, 0)
        
        vis_frames.append(overlay)
        
    # Save video
    height, width, layers = vis_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 5.0, (width, height)) # 5 fps
    
    for frame in vis_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
    out.release()
    print(f"Saved visualization to {output_path}")

if __name__ == '__main__':
    main()
