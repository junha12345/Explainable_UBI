import os
import sys
import argparse
import subprocess
import shutil

def run_command(cmd, description):
    print(f"\n[Pipeline] Starting: {description}")
    print(f"Command: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"[Pipeline] Finished: {description}")
    except subprocess.CalledProcessError as e:
        print(f"\n[Pipeline] ERROR: {description} failed with exit code {e.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="End-to-End Accident Analysis Pipeline (YOLO + VideoMAE + VLM)")
    parser.add_argument('--video_path', type=str, required=True, help='Absolute path to input video file')
    parser.add_argument('--output_dir', type=str, default='/workspace/output_analysis', help='Directory to save all results')
    parser.add_argument('--api_key', type=str, default="YOUR_PERPLEXITY_API_KEY", help='Perplexity API Key')
    args = parser.parse_args()

    # Absolute paths
    video_path = os.path.abspath(args.video_path)
    base_output_dir = os.path.abspath(args.output_dir)
    video_name = os.path.basename(video_path)
    video_stem = os.path.splitext(video_name)[0]
    
    # 0. Setup Directories
    os.makedirs(base_output_dir, exist_ok=True)
    
    vis_obj_dir = os.path.join(base_output_dir, "vis_obj") # For YOLO/Attrib videos
    vlm_frame_dir = os.path.join(base_output_dir, "vlm_frames") # For extracted frames
    
    # Scripts Paths (Hardcoded based on workspace structure)
    WORKSPACE_ROOT = "/workspace"
    SCRIPT_YOLO = os.path.join(WORKSPACE_ROOT, "VideoMAEv2", "run_yolo.py")
    SCRIPT_ATTRIB = os.path.join(WORKSPACE_ROOT, "VideoMAEv2", "object_attribution.py")
    SCRIPT_PREPARE_VLM = os.path.join(WORKSPACE_ROOT, "VLM_Project", "prepare_vlm_input.py")
    SCRIPT_ANALYZE_VLM = os.path.join(WORKSPACE_ROOT, "VLM_Project", "analyze_context.py")

    # ---------------------------------------------------------
    # Step 1: Run YOLO Tracking (Majority Voting & Labeling)
    # ---------------------------------------------------------
    # Output: yolo_only_{name}.mp4 and yolo_only_{name}.json
    
    cmd_step1 = f"python3 {SCRIPT_YOLO} --video_path {video_path} --num_frames -1 --output_dir {vis_obj_dir}"
    run_command(cmd_step1, "Step 1: YOLO Tracking & Consistency")
    
    # Expected Outputs from Step 1
    yolo_video_path = os.path.join(vis_obj_dir, f"yolo_only_{video_name}")
    yolo_json_path = os.path.join(vis_obj_dir, f"yolo_only_{video_stem}.json")
    
    if not os.path.exists(yolo_json_path):
        print(f"Error: Expected JSON output not found at {yolo_json_path}")
        # Try fallback matching if logic differs
        pass

    # ---------------------------------------------------------
    # Step 2: Run Object Attribution (VideoMAE + Risk + Heatmap)
    # ---------------------------------------------------------
    # Uses the JSON from Step 1 to sync IDs.
    
    cmd_step2 = f"python3 {SCRIPT_ATTRIB} --video_path {video_path} --detections_json {yolo_json_path} --output_dir {vis_obj_dir}"
    run_command(cmd_step2, "Step 2: VideoMAE Object Attribution")
    
    # ---------------------------------------------------------
    # Step 3: Extract Frames for VLM
    # ---------------------------------------------------------
    # INPUT: The YOLO-labeled video from Step 1 (so VLM sees IDs like 'car-4')
    # Sampling: Every 10 frames
    
    # Check if YOLO video exists
    if not os.path.exists(yolo_video_path):
        print(f"Error: YOLO video not found at {yolo_video_path}")
        sys.exit(1)
        
    cmd_step3 = f"python3 {SCRIPT_PREPARE_VLM} --video_path {yolo_video_path} --output_dir {vlm_frame_dir} --step 10"
    run_command(cmd_step3, "Step 3: VLM Frame Extraction")
    
    # Output dir for frames will be {vlm_frame_dir}/{yolo_only_video_name_no_ext}
    extracted_frames_path = os.path.join(vlm_frame_dir, f"yolo_only_{video_stem}")
    
    # ---------------------------------------------------------
    # Step 4: Run VLM Context Analysis
    # ---------------------------------------------------------
    
    cmd_step4 = f"python3 {SCRIPT_ANALYZE_VLM} --frame_dir {extracted_frames_path} --api_key {args.api_key}"
    run_command(cmd_step4, "Step 4: VLM Context Analysis")
    
    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("       PIPELINE EXECUTION COMPLETE")
    print("="*50)
    print(f"1. Tracking Video:   {yolo_video_path}")
    print(f"2. Analysis Video:   {os.path.join(vis_obj_dir, f'obj_attrib_{video_name}')}")
    print(f"3. Tracking Data:    {yolo_json_path}")
    print(f"4. Attribution Data: {os.path.join(vis_obj_dir, f'obj_attrib_{video_name}.json')}")
    print(f"5. VLM Analysis:     {os.path.join(extracted_frames_path, 'vlm_analysis.json')}")
    print("="*50)

if __name__ == "__main__":
    main()
