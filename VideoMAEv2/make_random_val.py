import os
import random
import csv

# Paths
TRAIN_LABEL_CSV = '/workspace/train_label.csv'
OUTPUT_CSV = '/workspace/VideoMAEv2/data/custom/val_random.csv'
VIDEO_ROOT = '/workspace/train'

# 1. Load All Labels
all_data = []
with open(TRAIN_LABEL_CSV, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)  # Skip header: id,target
    for row in reader:
        if len(row) >= 2:
            vid_id = row[0]
            label = int(float(row[1])) # Handle 0.0 -> 0
            
            # Construct full path
            # Assuming file extension is .mp4 (based on previous ls)
            # If ID doesn't have extension, add it.
            if not vid_id.endswith('.mp4'):
                filename = f"{vid_id}.mp4"
            else:
                filename = vid_id
            
            full_path = os.path.join(VIDEO_ROOT, filename)
            
            # Verify file exists
            if os.path.exists(full_path):
                all_data.append((full_path, label))
            else:
                # Try adding .mp4 if it failed
                 if os.path.exists(full_path + ".mp4"):
                     all_data.append((full_path + ".mp4", label))

print(f"Loaded {len(all_data)} valid samples from train_label.csv")

# 2. Randomly Sample 300
if len(all_data) > 300:
    selected_data = random.sample(all_data, 300)
else:
    selected_data = all_data
    print("Warning: Less than 300 samples available, using all.")

# 3. Write Output (Space Separated for VideoMAE dataset)
with open(OUTPUT_CSV, 'w') as f:
    for path, label in selected_data:
        f.write(f"{path} {label}\n")

print(f"Successfully created {OUTPUT_CSV} with {len(selected_data)} samples.")
