
import os
import sys
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from timm.models import create_model

# Add current directory to path so we can import models and dataset
sys.path.append(os.getcwd())

import models
from dataset import video_transforms, volume_transforms
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser('VideoMAE Prediction')
    parser.add_argument('--checkpoint', default='work_dirs/custom_finetune/checkpoint-19.pth', type=str)
    parser.add_argument('--video_path', required=True, type=str)
    parser.add_argument('--model', default='vit_base_patch16_224', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--num_frames', default=16, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    return parser.parse_args()

def load_video_decord(sample, num_frames=16):
    from decord import VideoReader, cpu
    vr = VideoReader(sample, ctx=cpu(0))
    length = len(vr)
    all_index = np.linspace(0, length-1, num_frames).astype(int)
    vr.seek(0)
    buffer = vr.get_batch(all_index).asnumpy()
    return buffer

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
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
    else:
        print(f"Checkpoint not found at {args.checkpoint}")
        return

    model.to(device)
    model.eval()

    # Load Video
    try:
        frames = load_video_decord(args.video_path, num_frames=args.num_frames)
    except Exception as e:
        print(f"Error loading video {args.video_path}: {e}")
        return

    # Transform
    transform_pipeline = video_transforms.Compose([
        video_transforms.Resize(256, interpolation='bilinear'),
        video_transforms.CenterCrop(size=(args.input_size, args.input_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    pil_frames = [Image.fromarray(f) for f in frames]
    data = transform_pipeline(pil_frames) # (C, T, H, W)
    data = data.unsqueeze(0).to(device) # (1, C, T, H, W)

    # Predict
    with torch.no_grad():
        logits = model(data)
        probs = F.softmax(logits, dim=1)
        
    probs = probs.cpu().numpy()[0]
    
    # Assuming Class 1 = Accident, Class 0 = Normal based on typical binary classification
    # Modify labels as needed if you find definitive mapping
    print(f"Video: {os.path.basename(args.video_path)}")
    print(f"Probabilities: No Accident (0): {probs[0]:.4f}, Accident (1): {probs[1]:.4f}")
    if probs[1] > probs[0]:
        print("Prediction: Accident detected")
    else:
        print("Prediction: No accident detected")

if __name__ == '__main__':
    main()
