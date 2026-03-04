
import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from decord import VideoReader, cpu
import json
from timm.models import create_model

# Add current directory to path
sys.path.append(os.getcwd())

import models
from dataset import video_transforms, volume_transforms

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics not installed. Please run 'pip install ultralytics'")
    sys.exit(1)

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
        
        # Calculate probabilities
        probs = F.softmax(logits, dim=1)
        
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
            
        # Get target probability
        target_prob = probs[0, class_idx].item()
        
        # Target score
        one_hot = torch.zeros_like(logits)
        one_hot[0][class_idx] = 1
        
        # Backward
        logits.backward(gradient=one_hot, retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        
        # Global Average Pooling of gradients
        weights = torch.mean(gradients, dim=1, keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(activations * weights, dim=2)
        
        # ReLU
        cam = F.relu(cam)
        
        return cam, class_idx, target_prob

def get_args():
    parser = argparse.ArgumentParser('VideoMAE Object Attribution')
    parser.add_argument('--checkpoint', default='work_dirs/custom_finetune/checkpoint-19.pth', type=str)
    parser.add_argument('--video_path', required=True, type=str)
    parser.add_argument('--output_dir', default='test_video_vis_obj', type=str)
    parser.add_argument('--model', default='vit_base_patch16_224', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--num_frames', default=16, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--target_class', default=None, type=int)
    parser.add_argument('--yolo_model', default='yolo11x.pt', type=str, help='YOLO model to use')
    parser.add_argument('--conf_thres', default=0.15, type=float, help='YOLO confidence threshold')
    parser.add_argument('--show_heatmap', action='store_true', help='Overlay Grad-CAM heatmap on video')
    parser.add_argument('--detections_json', default=None, type=str, help='Path to master detection JSON for ID sync')
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
    B, N = tensor.shape
    expected = t_feat * h_feat * w_feat
    
    if N == expected + 1:
        tensor = tensor[:, 1:]
    
    result = tensor.reshape(B, t_feat, h_feat, w_feat)
    return result

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading VideoMAE model: {args.model}")
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
        print("VideoMAE Model loaded.")
    else:
        print("Checkpoint not found.")
        return

    model.to(device)
    model.eval()
    
    # Target Layer
    target_layer = model.blocks[-1]
    grad_cam = GradCAM(model, target_layer)
    
    print(f"Loading YOLO model: {args.yolo_model}")
    yolo = YOLO(args.yolo_model)
    
    # Load Video
    # frames, _ = load_video_decord(args.video_path, num_frames=args.num_frames)
    
    # Load video with Decord to get metadata
    vr = VideoReader(args.video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    duration = len(vr) / fps
    total_video_frames = len(vr)
    
    # Load ALL frames for Visualization/YOLO stability
    # But keep VideoMAE index separate (16 frames)
    all_frames_idx = np.arange(total_video_frames)
    mae_sample_idx = np.linspace(0, total_video_frames-1, args.num_frames).astype(int)
    
    vr.seek(0)
    # Load all frames for VIS
    # Memory warning: 150 frames 640x640 is fine. 
    frames = vr.get_batch(all_frames_idx).asnumpy()
    
    # Load Master Detections if provided
    master_detections = {}
    if args.detections_json:
        with open(args.detections_json, 'r') as f:
            data = json.load(f)
            # Index by frame_idx for O(1) lookup
            for d in data.get('detections', []):
                master_detections[d['frame_idx']] = d['objects']
        print(f"Loaded master detections from {args.detections_json}")
    
    # Initialize JSON output
    json_detections = []
    json_output = {
        "video_duration": duration,
        "fps": fps,
        "detections": json_detections
    }
    
    # Extract frames for MAE (16 frames)
    # We can just pick them from the loaded frames array using mapping
    # But frames array is huge, MAE needs 224x224.
    
    # Transform for VideoMAE
    transform_pipeline = video_transforms.Compose([
        video_transforms.Resize(256, interpolation='bilinear'),
        video_transforms.CenterCrop(size=(args.input_size, args.input_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert numpy frames to PIL for transformation
    pil_frames = [Image.fromarray(f) for f in frames]
    
    # Create MAE Input from sampled frames
    mae_pil_frames = [pil_frames[i] for i in mae_sample_idx]
    data = transform_pipeline(mae_pil_frames)
    data = data.unsqueeze(0).to(device)
    data.requires_grad = True
    
    # Run Grad-CAM (on 16 frames)
    cam, pred_class, pred_prob = grad_cam(data, class_idx=args.target_class)
    
    val = pred_class.item() if isinstance(pred_class, torch.Tensor) else pred_class
    print(f"Prediction Class: {val}, Probability: {pred_prob:.4f}")
    
    # Add prediction to JSON
    json_output["model_prediction_class"] = int(val)
    json_output["model_prediction_prob"] = float(pred_prob)
    
    # Reshape CAM (16, H, W)
    t_feat = args.num_frames // 2
    h_feat = args.input_size // 16
    w_feat = args.input_size // 16
    
    cam = reshape_transform(cam, t_feat, h_feat, w_feat)
    cam = cam[0].cpu().detach().numpy() # (T_half, H, W) -> usually 8 frames for 16 input
    
    # VideoMAE usually works with Tubelet size 2, so output temporal dim is T/2.
    # We need to interpolate this back to full video length (150 frames).
    
    # Normalize CAM globally first
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    # Interpolate CAM to Full Length
    # cam: (8, H, W) -> (150, H, W)
    # We can use scipy or simple torch interpolation
    cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0) # (1, 1, 8, H, W)
    # Interpolate temporal dimension
    cam_resized = F.interpolate(cam_tensor, size=(total_video_frames, cam.shape[1], cam.shape[2]), mode='trilinear', align_corners=False)
    cam_resized = cam_resized.squeeze().numpy() # (150, H, W)
    
    # Visualization High-Res
    # Target resolution for visualization and YOLO
    vis_size = 640 
    
    os.makedirs(args.output_dir, exist_ok=True)
    video_name = os.path.basename(args.video_path)
    output_path = os.path.join(args.output_dir, f"obj_attrib_{video_name}")
    
    # Transform for Visualization (High Res Center Crop)
    base_resize = int(vis_size * (256/224))
    
    vis_transform = video_transforms.Compose([
        video_transforms.Resize(base_resize),
        video_transforms.CenterCrop(vis_size)
    ])
    
    vis_frames = []
    
    # Loop over ALL frames
    for i in range(total_video_frames):
        # Prepare frame for VIS/YOLO
        orig_img = pil_frames[i]
        vis_img = vis_transform([orig_img])[0] 
        vis_np = np.array(vis_img) # (640, 640, 3)
        
        # Get CAM for this frame (Interpolated)
        cam_map = cam_resized[i]
        
        # Resize CAM to VIS size (640x640)
        cam_map_resized = cv2.resize(cam_map, (vis_size, vis_size), interpolation=cv2.INTER_CUBIC)
        
        # Object Detection source: Master JSON or Live YOLO
        boxes = [] # List of box-like objects
        
        class MockBox:
            def __init__(self, obj_data):
                self.xyxy = torch.tensor([obj_data['box']]) # Mock tensor shape (1, 4)
                self.cls = torch.tensor([0]) # Mock class tensor
                # We need to hack cls_name lookup. 
                # yolo.names is {0: 'person', 1: 'bicycle', ...}
                # We have 'class': 'car'. 
                # We can attach the name directly and modify usage slightly or reverse lookup.
                self.cls_name_direct = obj_data['class']
                self.id = torch.tensor([obj_data['id']])
                self.conf = torch.tensor([0.9]) # Dummy confidence

        if args.detections_json:
            # Load from JSON
            current_frame_idx = all_frames_idx[i]
            objs = master_detections.get(current_frame_idx, [])
            
            for o in objs:
                boxes.append(MockBox(o))
                
        else:
            # Run YOLO Tracking (Legacy mode)
            results = yolo.track(vis_np, verbose=False, conf=args.conf_thres, iou=0.5, persist=True, tracker="bytetrack.yaml")
            if results[0].boxes is not None:
                boxes = results[0].boxes
        
        max_score = -1.0
        best_box = None
        

        
        # Calculate scores
        # Calculate scores
        # Calculate scores
        box_data = [] # List of (score, x1, y1, x2, y2, cls_name, box_obj)
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Clip to image bounds
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(vis_size, x2); y2 = min(vis_size, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            # Extract class name first
            if hasattr(box, 'cls_name_direct'):
                cls_name = box.cls_name_direct
            else:
                cls_name = yolo.names[int(box.cls[0])]
            
            # Filter classes
            ALLOWED_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
            if cls_name not in ALLOWED_CLASSES:
                continue

            # Set size threshold based on class
            # Person: must be > 1/25
            # Others: must be > 1/15
            if cls_name == 'person':
                threshold = vis_size / 25.0
            else:
                threshold = vis_size / 15.0
                
            # Filter small objects
            box_w = x2 - x1
            box_h = y2 - y1
            
            # Must be larger than threshold in BOTH dimensions
            if box_w < threshold or box_h < threshold:
                continue
                
            # Get mean attention in the box
            roi = cam_map_resized[y1:y2, x1:x2]
            score = np.sum(roi)
            
            # Filter low-attention objects
            # If the total attention sum is very small, it's just noise.
            # Threshold 20.0 is arbitrary but reasonable for 640x640 sum of [0,1] scores.
            if score < 20.0:
                continue
            
            box_data.append({
                'raw_score': score,
                'coords': (x1, y1, x2, y2),
                'cls_name': cls_name,
                'box': box
            })

        # Apply Softmax if boxes exist
        # Apply Softmax if boxes exist
        if box_data:
            raw_scores = np.array([b['raw_score'] for b in box_data])
            
            # Normalize raw_scores to a reasonable range [0, 1] before softmax
            # Because 'Sum' can be large (e.g. 500), exp(500) causes overflow/saturation.
            
            if len(raw_scores) > 1:
                score_min = raw_scores.min()
                score_max = raw_scores.max()
                
                # Avoid division by zero
                if score_max - score_min > 1e-6:
                    # Normalize to [0, 1]
                    norm_scores = (raw_scores - score_min) / (score_max - score_min)
                    
                    # Apply temperature to control sharpness (Higher = Sharper, Lower = Softer)
                    # 5.0 gives good separation without being binary
                    temperature = 5.0 
                    scaled_scores = norm_scores * temperature
                else:
                    # All scores are same
                    scaled_scores = np.zeros_like(raw_scores)
            else:
                 scaled_scores = np.zeros_like(raw_scores)

            exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
            softmax_scores = exp_scores / exp_scores.sum()
            
            # Update scores in box_data
            # Update scores in box_data
            for idx, b in enumerate(box_data):
                b['softmax_score'] = softmax_scores[idx]
            
            # Calculate Risk Scores for ALL boxes first
            # Weight w(t) = P * (1 - exp(-5 * t/T))
            if total_video_frames > 0:
                time_progression = i / float(total_video_frames)
                time_weight = pred_prob * (1.0 - np.exp(-5.0 * time_progression))
            else:
                time_weight = pred_prob
                
            for b in box_data:
                # Magnitude Weight: Scale down risk if attention sum is low relative to a strong signal
                # Saturation point = 1000.0 (User adjusted)
                magnitude_weight = min(1.0, b['raw_score'] / 1000.0)
                
                # Final Risk = Softmax (Relative) * Time (Temporal) * Magnitude (Absolute)
                b['risk_score'] = b['softmax_score'] * time_weight * magnitude_weight
                
            # Find best box based on FINAL RISK SCORE
            # This ensures the highlight red box logic follows the user's request strictly
            best_idx = np.argmax([b['risk_score'] for b in box_data])
            best_candidate = box_data[best_idx]
            
            # User Request: Only highlight Red if Risk Score >= 70%
            if best_candidate['risk_score'] >= 0.7:
                best_box_data = best_candidate
            else:
                best_box_data = None
            
        else:
            best_box_data = None


        frame_vis = vis_np.copy()

        if args.show_heatmap:
            heatmap = cm.jet(cam_map_resized)[..., :3] * 255.0
            heatmap = heatmap.astype(np.uint8)
            alpha = 0.4
            frame_vis = cv2.addWeighted(frame_vis, 1-alpha, heatmap, alpha, 0)


            
        # Draw Boxes
        for idx, b in enumerate(box_data):
            x1, y1, x2, y2 = b['coords']
            
            # Use pre-calculated Risk Score
            score_pct = b['risk_score'] * 100
            
            cls_name = b['cls_name']
            
            # Default: Green, Thickness 2
            color = (0, 255, 0)
            thickness = 2
            
            # Check if this is the best box
            if b == best_box_data:
                color = (255, 0, 0) # Red for the "cause" (Same thickness, just different color)
                
            label = f"{cls_name} {score_pct:.1f}%"
            
            # Draw Label slightly above box
            cv2.putText(frame_vis, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, thickness)
            
        # Collect JSON data for this frame
        frame_timestamp = i * (duration / total_video_frames) # Correct timestamp for each frame
        frame_objects = []
        
        for b in box_data:
            obj_id = int(b['box'].id[0]) if b['box'].id is not None else -1
            frame_objects.append({
                "class": b['cls_name'],
                "id": obj_id,
                "risk_score": float(b['risk_score']), # 0.0 - 1.0
                "attention_sum": float(b['raw_score'])
            })
            
        json_detections.append({
            "timestamp": round(frame_timestamp, 3),
            "frame": int(all_frames_idx[i]), # Original frame index
            "objects": frame_objects
        })
            
        vis_frames.append(frame_vis)

    # Save JSON
    json_path = os.path.join(args.output_dir, f"obj_attrib_{video_name}.json")
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=4)
    print(f"Saved Attribution JSON to {json_path}")

    # Save Video
    height, width, layers = vis_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 5.0, (width, height))
    
    for frame in vis_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) # OpenCV expects BGR
    out.release()
    print(f"Saved Object Attribution Video to {output_path}")

if __name__ == '__main__':
    main()
