
import os
import glob
import base64
import json
import argparse
import requests
from openai import OpenAI

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_args():
    parser = argparse.ArgumentParser(description='Analyze frames with VLM')
    parser.add_argument('--frame_dir', type=str, required=True, help='Directory containing extracted frames')
    parser.add_argument('--api_key', type=str, default=os.environ.get("OPENAI_API_KEY"), help='API Key (Perplexity/OpenAI/etc)')
    parser.add_argument('--base_url', type=str, default="https://api.perplexity.ai", help='API Base URL')
    parser.add_argument('--model', type=str, default="sonar", help='Model name')
    return parser.parse_args()

def main():
    args = get_args()
    
    if not args.api_key:
        print("Error: API Key is required. Set API_KEY env var or pass --api_key")
        # For demo purposes, we will just print what would happen
        # return

    # Gather frames
    frame_files = sorted(glob.glob(os.path.join(args.frame_dir, "*.jpg")))
    if not frame_files:
        print(f"No frames found in {args.frame_dir}")
        return

    print(f"Found {len(frame_files)} frames for analysis.")
    
    # Prepare Message Payload
    content = []
    content.append({
        "type": "text", 
        "text": f"""
        You are an expert Traffic Accident Analysis AI.
        You are provided with a sequence of video frames from a driving video.
        IMPORTANT: These frames are sampled at a rate of 1 frame every 10 frames (approx. 0.4 seconds interval).
        - Frame 0 corresponds to the original video frame 0.
        - Frame 1 corresponds to the original video frame 10.
        - Frame 2 corresponds to the original video frame 20.
        And so on. Total original duration is approx. 150 frames.
        
        Objects are detected with Bounding Boxes and Tracking IDs (e.g., 'car-1', 'person-3').
        
        Task:
        1. Analyze the overall situation and context of the road.
        2. Track the movement of specific objects based on their IDs across these sampled frames.
        3. Identify the status of Traffic Lights. Be specific about Vehicle vs Pedestrian lights.
        4. Check if the vehicle directly in front has its Brake Lights ON.
        
        Output must be strictly valid JSON with no markdown formatting:
        {{
            "overall_situation": "Summary of road type, weather, visible lanes...",
            "vehicle_traffic_light_status": "Red/Green/Yellow/Unknown",
            "pedestrian_traffic_light_status": "Red/Green/Unknown (or Not Visible)",
            "front_vehicle_brake_light_status": "On/Off/Not Visible/No Vehicle Ahead",
            "sudden_events": ["List of sudden appearances..."],
            "object_movements": [
                {{
                    "id": "car-1",
                    "description": "Moved from left lane to center..."
                }},
                ...
            ]
        }}
        """
    })

    for f in frame_files:
        base64_image = encode_image(f)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    # Client Setup (Example using OpenAI SDK which is compatible with many providers)
    if args.api_key:
        client = OpenAI(api_key=args.api_key, base_url=args.base_url)
        
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "user", "content": content}
                ],
                max_tokens=4096,
                # Perplexity API does not support response_format="json_object"
                # response_format={ "type": "json_object" }
            )
            
            result = response.choices[0].message.content
            print(result)
            
            # Clean up potential markdown formatting
            cleaned_result = result.replace("```json", "").replace("```", "").strip()
            
            # Save to file with pretty printing
            out_path = os.path.join(args.frame_dir, "vlm_analysis.json")
            try:
                # Try parsing as JSON to format it
                json_data = json.loads(cleaned_result)
                with open(out_path, "w", encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                # Fallback to raw string if parsing fails
                print("Warning: Could not parse response as JSON. Saving raw output.")
                with open(out_path, "w") as f:
                    f.write(cleaned_result)
            
            print(f"Saved analysis to {out_path}")
            
        except Exception as e:
            print(f"API Request Failed: {e}")
    else:
        print("Mock Run: Frames prepared, prompt constructed. Set --api_key to execute.")

if __name__ == "__main__":
    main()
