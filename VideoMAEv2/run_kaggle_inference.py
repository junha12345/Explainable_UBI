
import argparse
import numpy as np
import torch
import pandas as pd
import os
from timm.models import create_model
from torchvision import transforms
from dataset import video_transforms, volume_transforms
from decord import VideoReader, cpu
import models # Register VideoMAE models

def get_args():
    parser = argparse.ArgumentParser(description='VideoMAE V2 Kaggle Inference')
    parser.add_argument('--test_dir', type=str, default='/workspace/kaggle_data/test', help='Directory containing test videos')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to fine-tuned checkpoint')
    parser.add_argument('--output_csv', type=str, default='submission.csv', help='Where to save the result')
    parser.add_argument('--model', type=str, default='vit_base_patch16_224')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device(args.device)

    # 1. Model Setup
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=2,  # Binary: 0=Safe, 1=Accident
        all_frames=args.num_frames,
        tubelet_size=2,
        use_mean_pooling=True
    )

    # 2. Load Checkpoint (Safe Load)
    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove DDP prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()

    # 3. Prepare Transform
    data_transform = video_transforms.Compose([
        video_transforms.Resize(224, interpolation='bilinear'),
        video_transforms.CenterCrop(size=(224, 224)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Process Videos
    results = []
    
    # Check if test dir exists (it might not yet)
    if not os.path.exists(args.test_dir):
        print(f"Warning: Test directory {args.test_dir} not found yet. Please assume it will be there.")
        # Mock logic or empty list
        video_files = []
    else:
        video_files = [f for f in os.listdir(args.test_dir) if f.endswith('.mp4')]
        video_files.sort() # Ensure consistent order

    print(f"Found {len(video_files)} videos in {args.test_dir} (if downloaded).")

    with torch.no_grad():
        for vid_file in video_files:
            vid_path = os.path.join(args.test_dir, vid_file)
            try:
                # Load Video
                vr = VideoReader(vid_path, ctx=cpu(0))
                total_frames = len(vr)
                indices = np.linspace(0, total_frames - 1, args.num_frames).astype(int)
                images = vr.get_batch(indices).asnumpy()

                # Transform
                pil_imgs = [transforms.ToPILImage()(img) for img in images]
                input_tensor = data_transform(pil_imgs)
                input_tensor = input_tensor.unsqueeze(0).to(device)

                # Inference
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                # Class 1 is "Accident"
                p_accident = probs[1]
                
                # Kaggle output: id, prediction
                # id is usually filename without extension? Check submission sample.
                # Assuming id = filename (e.g. 00001.mp4 or just 00001)
                vid_id = os.path.splitext(vid_file)[0] 
                
                results.append({
                    'id': vid_id,
                    'target': p_accident  # Probability of accident
                })
                
                if len(results) % 10 == 0:
                    print(f"Processed {len(results)} videos...")

            except Exception as e:
                print(f"Error processing {vid_file}: {e}")

    # 5. Save Submission
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output_csv, index=False)
        print(f"Submission saved to {args.output_csv} with {len(results)} rows.")
    else:
        print("No results generated (folder might be empty).")

if __name__ == '__main__':
    main()
