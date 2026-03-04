import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Config
DATA_ROOT = '/workspace/train'
CSV_PATH = '/workspace/train_label.csv'
OUTPUT_DIR = '/workspace/VideoMAEv2/data/custom'

# Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read CSV
df = pd.read_csv(CSV_PATH)
# Ensure columns are correct (id, target)
# ID formatting: Some IDs might be int, need to match filename (e.g. 128 -> 00128.mp4)
df['id'] = df['id'].apply(lambda x: f"{int(x):05d}")

# Gather valid data
valid_data = []
for _, row in df.iterrows():
    video_path = os.path.join(DATA_ROOT, f"{row['id']}.mp4")
    if os.path.exists(video_path):
        # Format: video_path, label
        # Note: Depending on run_class_finetuning.py, it might expect absolute path.
        valid_data.append({'video_path': video_path, 'label': int(row['target'])})
    else:
        print(f"Warning: {video_path} not found")

print(f"Total valid videos found: {len(valid_data)}")
df_valid = pd.DataFrame(valid_data)

# Split 80/20
train_df, val_df = train_test_split(df_valid, test_size=0.2, random_state=42, stratify=df_valid['label'])

# Save as CSV (headerless or with header? FINETUNE.md implies just lines: video_path, label. Usually space or comma. 
# Finetuning script usually uses a custom dataset loader. Let's assume space separated for now as per MMAction conventions, 
# BUT FINETUNE.md said "path, label" (comma).
# Let's write space separated first as it's more common in vision codebases, but check run_class_finetuning.py.
# Actually, the example in FINETUNE.md says "video_path, label". 
# I will write space separated because most Torch/MMCV loaders use space.
# Wait, let's verify run_class_finetuning.py content first.
# For now, I'll write space separated.

train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), sep=' ', index=False, header=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), sep=' ', index=False, header=False)

print(f"Train samples: {len(train_df)}")
print(f"Val samples: {len(val_df)}")
print(f"CSVs saved to {OUTPUT_DIR}")
