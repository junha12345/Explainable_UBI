import argparse
import numpy as np
import torch
import pandas as pd
import os
from mmcv.runner import load_checkpoint
from timm.models import create_model
from dataset.datasets import VideoClsDataset
from torch.utils.data import DataLoader
from functools import partial
from utils import multiple_samples_collate

# Import necessary modules from the repo (we assume we run this from inside VideoMAEv2 dir)
import utils
import models # Registers models

def get_args():
    parser = argparse.ArgumentParser(description='VideoMAE V2 Inference and Scoring')
    parser.add_argument('--video_path', type=str, required=True, help='Path to video file or directory of videos')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to fine-tuned checkpoint')
    parser.add_argument('--model', type=str, default='vit_base_patch16_224', help='Model architecture')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--nb_classes', type=int, default=2)
    parser.add_argument('--output_csv', type=str, default='inference_results.csv')
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device)

    # 1. Load Model
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames * 1, # num_segments=1
        tubelet_size=2,
        use_mean_pooling=True
    )
    
    # Load checkpoint
    # Custom load to handle PyTorch 2.6 pickle security
    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Handle state dict format (sometimes wrapped in 'model' or 'state_dict')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if ddp was used
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()

    # 2. Prepare Data
    # We can reuse VideoClsDataset logic but slightly modified for inference (no label needed ideally, but the class requires it)
    # Or just use a simple custom loader using the same transforms.
    # For simplicity, let's create a temporary CSV if input is a directory.
    
    video_paths = []
    if os.path.isdir(args.video_path):
        for f in os.listdir(args.video_path):
            if f.endswith('.mp4'):
                video_paths.append(os.path.join(args.video_path, f))
    else:
        video_paths.append(args.video_path)

    # Define transforms (matches validation transform in datasets.py)
    from torchvision import transforms
    from dataset import video_transforms, volume_transforms
    
    # Transforms
    data_transform = video_transforms.Compose([
        video_transforms.Resize(224, interpolation='bilinear'),
        video_transforms.CenterCrop(size=(224, 224)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Decord Loader
    from decord import VideoReader, cpu
    
    results = []
    
    print(f"Starting inference on {len(video_paths)} videos...")
    
    with torch.no_grad():
        for vid_path in video_paths:
            try:
                vr = VideoReader(vid_path, ctx=cpu(0))
                # Sampling logic (Center sampling or uniform)
                # Let's match training: uniform sampling
                # But for inference, we usually want deterministic. 
                # Let's simple uniform sample
                total_frames = len(vr)
                
                # frame_sample_rate=8, num_frames=16 -> covers 128 frames.
                # If video is shorter, we loop.
                # If longer, we center crop temporally? Or just start from 0 if it's ~145 frames.
                # Let's just simply take linspace
                
                indices = np.linspace(0, total_frames - 1, args.num_frames).astype(int)
                images = vr.get_batch(indices).asnumpy() # T H W C
                
                # Transform
                # T H W C -> T C H W (ToTensor handles this? No, dataset does ToPIL then ToTensor)
                # dataset.py: [transforms.ToPILImage()(frame) for frame in buffer]
                # data_transform expects list of PIL images or Tensor?
                # looking at validation_one_epoch in dataset.py:
                # buffer = self.data_transform(buffer)
                # buffer comes from load_video which returns T H W C numpy
                # Wait, datasets.py: _aug_frame uses PIL. But validation path uses video_transforms directly?
                # validation path: buffer = self.load_video... buffer = self.data_transform(buffer)
                # let's look at validation transform: video_transforms.Compose...
                # ClipToTensor expects list of H W C numpy? or list of PIL? 
                
                # Let's replicate strict logic from datasets.py for validation
                # It seems volume_transforms.ClipToTensor() handles list of arrays or list of PILs
                # Let's convert to list of PILs to be safe as per dataset.py
                
                pil_imgs = [transforms.ToPILImage()(img) for img in images]
                input_tensor = data_transform(pil_imgs) # C T H W
                
                # Add batch dim
                input_tensor = input_tensor.unsqueeze(0).to(device)
                
                # Inference
                logits = model(input_tensor) # [1, 2]
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                # Assuming Class 0 = Safe, Class 1 = Accident
                # Wait, we need to verify class mapping. 
                # Usually sorted by alphabet or appeared order.
                # train_label.csv: 0.0, 1.0. 
                # target=1.0 is Accident? Usually yes.
                # Let's assume index 1 is Accident (p).
                
                p_accident = probs[1]
                p_safe = probs[0]
                
                # Driver Fault Score = (1 - p) * 100
                # If p_accident is high (0.9) -> Score = 10 (Low Fault) -> "Unavoidable"
                # If p_accident is low (0.1) -> Score = 90 (High Fault) -> "Should have been safe"
                
                score = (1 - p_accident) * 100
                
                results.append({
                    'video': os.path.basename(vid_path),
                    'p_accident': p_accident,
                    'p_safe': p_safe,
                    'driver_fault_score': score
                })
                
                print(f"Processed {os.path.basename(vid_path)}: p_accident={p_accident:.4f}, Score={score:.2f}")

            except Exception as e:
                print(f"Error processing {vid_path}: {e}")
                
    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")

if __name__ == '__main__':
    main()
