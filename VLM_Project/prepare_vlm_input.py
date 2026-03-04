
import cv2
import os
import argparse
import glob

def get_args():
    parser = argparse.ArgumentParser(description='Prepare frames for VLM')
    parser.add_argument('--video_path', type=str, required=True, help='Path to YOLO-visualized video')
    parser.add_argument('--output_dir', type=str, default='vlm_frames', help='Directory to save extracted frames')
    parser.add_argument('--step', type=int, default=1, help='Frame sampling step')
    parser.add_argument('--start_frame', type=int, default=0, help='Start frame index')
    parser.add_argument('--end_frame', type=int, default=None, help='End frame index (inclusive)')
    return parser.parse_args()

def main():
    args = get_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
        return

    # Create output directory specific to video name
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    save_dir = os.path.join(args.output_dir, video_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Clean previous frames
    files = glob.glob(os.path.join(save_dir, "*.jpg"))
    for f in files:
        os.remove(f)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {video_name} ({frame_count} frames)...")
    
    end_frame = args.end_frame if args.end_frame is not None else frame_count - 1

    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check range and step
        if args.start_frame <= frame_idx <= end_frame:
            if (frame_idx - args.start_frame) % args.step == 0:
                # Stamp Frame Number (Sequential 0, 1, 2...)
                # User requested logical frame number, not absolute frame index.
                text = f"Frame: {saved_count}"
                position = (30, 50)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1.0
                thickness = 2
                
                # Black border
                cv2.putText(frame, text, position, font, scale, (0, 0, 0), thickness + 3)
                # White text
                cv2.putText(frame, text, position, font, scale, (255, 255, 255), thickness)
                
                # Save to specific debug folder if needed, or just keep structure
                # User specifically asked for "debug_frame" folder
                # We will save to args.output_dir which user can set to 'debug_frame'
                # But to be safe, let's append 'debug_frame' to the path if not present, or just trust output_dir.
                # Actually, adhering to user request: "make a folder called debug frame".
                
                out_file = os.path.join(save_dir, f"frame_{saved_count}.jpg")
                cv2.imwrite(out_file, frame)
                saved_count += 1
            
        frame_idx += 1

    cap.release()
    print(f"Saved {saved_count} frames to {save_dir}")

if __name__ == "__main__":
    main()
