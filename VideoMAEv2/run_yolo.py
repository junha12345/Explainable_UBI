
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
import os
from PIL import Image
from decord import VideoReader, cpu
from ultralytics import YOLO

# Add current directory to path
sys.path.append(os.getcwd())

# Import transforms (Assuming they are available in dataset)
import dataset.video_transforms as video_transforms
import dataset.volume_transforms as volume_transforms

def get_args():
    parser = argparse.ArgumentParser(description='VideoMAE YOLO Only Visualization')
    parser.add_argument('--video_path', default='', type=str, help='path to video')
    parser.add_argument('--yolo_model', default='yolo11x.pt', type=str, help='YOLO model checkpoint')
    parser.add_argument('--output_dir', default='test_video_vis_obj', type=str, help='output directory')
    parser.add_argument('--num_frames', default=16, type=int, help='frames to sample')
    parser.add_argument('--conf_thres', default=0.25, type=float, help='YOLO confidence threshold')
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print(f"Loading YOLO model: {args.yolo_model}")
    yolo = YOLO(args.yolo_model)
    
    # Load video with Decord to get metadata
    vr = VideoReader(args.video_path, ctx=cpu(0))
    total_video_frames = len(vr)
    
    # Load sampled frames
    if args.num_frames == -1:
        all_index = np.arange(total_video_frames)
        args.num_frames = total_video_frames
    else:
        all_index = np.linspace(0, total_video_frames-1, args.num_frames).astype(int)
        
    vr.seek(0)
    frames = vr.get_batch(all_index).asnumpy()
    
    # Visualization High-Res Target
    vis_size = 640 
    
    os.makedirs(args.output_dir, exist_ok=True)
    video_name = os.path.basename(args.video_path)
    output_path = os.path.join(args.output_dir, f"yolo_only_{video_name}")
    
    # Transform for Visualization (High Res Center Crop)
    # Matches object_attribution.py logic exactly
    base_resize = int(vis_size * (256/224))
    
    vis_transform = video_transforms.Compose([
        video_transforms.Resize(base_resize),
        video_transforms.CenterCrop(vis_size)
    ])
    
    pil_frames = [Image.fromarray(f) for f in frames]
    vis_frames = []
    
    # Initialize JSON data and Tracking History
    json_detections = []
    track_class_history = {} # {id: [class, class, ...]}
    
    # Store raw detections temporarily: [frame_idx, [(box, class, id), ...]]
    raw_frame_detections = []
    
    print(f"Processing {args.num_frames} frames (Pass 1: Detection)...")
    
    # Pass 1: Run YOLO and Collect Data
    for i in range(args.num_frames):
        # Prepare frame
        orig_img = pil_frames[i]
        vis_img = vis_transform([orig_img])[0] 
        vis_np = np.array(vis_img) # (640, 640, 3)
        
        # Run YOLO Tracking
        results = yolo.track(vis_np, verbose=False, conf=args.conf_thres, iou=0.5, persist=True, tracker="bytetrack.yaml")
        boxes = results[0].boxes
        
        current_frame_dets = []
        
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                cls_name = yolo.names[cls_id]
                track_id = int(box.id[0]) if box.id is not None else -1
                
                ALLOWED_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
                if cls_name not in ALLOWED_CLASSES:
                    continue

                # Class-Specific Confidence Thresholds to reduce False Positives
                # User reported reflections on cars being detected as people.
                # Increase Person threshold significantly.
                if cls_name == 'person' and box.conf[0] < 0.45:
                    continue
                    
                # For vehicles, keep it a bit looser but safer than 0.15
                # UPDATE: User reported buildings being misclassified as trucks (e.g. truck-51).
                # Increased threshold to 0.40 to remove these false positives.
                if cls_name != 'person' and box.conf[0] < 0.40:
                    continue
                
                dim_thresh = vis_size / 25 if cls_name == 'person' else vis_size / 15
                if (x2 - x1) < dim_thresh or (y2 - y1) < dim_thresh:
                    continue
                    
                # Collect class history
                if track_id != -1:
                    if track_id not in track_class_history:
                        track_class_history[track_id] = []
                    track_class_history[track_id].append(cls_name)
                    
                current_frame_dets.append({
                    "box": [x1, y1, x2, y2],
                    "class": cls_name,
                    "id": track_id
                })
        
        raw_frame_detections.append(current_frame_dets)

    # Resolve Majority Class per ID
    final_id_classes = {}
    from collections import Counter
    for tid, classes in track_class_history.items():
        if not classes: continue
        most_common = Counter(classes).most_common(1)[0][0]
        final_id_classes[tid] = most_common
        
    print("Consolidated Classes:", final_id_classes)
    
    # Pass 2: Draw and Save
    print("Generating Consistency Video (Pass 2: Drawing)...")
    
    for i in range(args.num_frames):
        # Re-create vis_np just to be safe (or use cached if memory permits, but fast enough to re-transform or just cache)
        # We didn't cache frame pixels to save memory, so re-transform.
        orig_img = pil_frames[i]
        vis_img = vis_transform([orig_img])[0] 
        vis_np = np.array(vis_img)
        frame_vis = vis_np.copy()
        
        # Get processed detections
        dets = raw_frame_detections[i]
        frame_objects = []
        
        for det in dets:
            track_id = det['id']
            # Apply Majority Class
            if track_id in final_id_classes:
                final_class = final_id_classes[track_id]
            else:
                final_class = det['class']
            
            x1, y1, x2, y2 = det['box']
            
            # Add to JSON output
            frame_objects.append({
                "class": final_class,
                "id": track_id,
                "box": [x1, y1, x2, y2]
            })
            
            # Draw
            color = (0, 255, 0) 
            thickness = 2
            label = f"{final_class}-{track_id}"
            
            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame_vis, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        json_detections.append({
            "frame_idx": int(all_index[i]),
            "objects": frame_objects
        })
        
        vis_frames.append(frame_vis)

    # Save Video
    height, width, layers = vis_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 5.0, (width, height)) # 5 FPS

    for f in vis_frames:
        f_bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        out.write(f_bgr)

    out.release()
    print(f"Saved YOLO Detection Video to {output_path}")
    
    # Save JSON
    json_path = output_path.replace('.mp4', '.json')
    import json
    with open(json_path, 'w') as f:
        json.dump({"detections": json_detections}, f, indent=4)
    print(f"Saved YOLO Detection JSON to {json_path}")

if __name__ == "__main__":
    main()
