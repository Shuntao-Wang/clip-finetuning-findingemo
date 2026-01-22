# prepare_subset.py
import os
import json
import shutil
import random
from collections import defaultdict
from findingemo_light.paper.download_multi import download_data
from findingemo_light.data.read_annotations import read_annotations

print("Step 1: Downloading FindingEmo pictures (This may take several hours; you can skip the downloaded section)...")
download_data(target_dir='./findingemo_full_images')

print("Step 2: Loading annotations...")
ann_data = read_annotations()  # dict: {filename: metadata}

# Extract the dominant emotion for each image (FindingEmo annotations typically include an 'emotion' field).
emotion_to_images = defaultdict(list)
for filename, meta in ann_data.items():
    # Since the dataset is frequently updated, adjust the following line according to the actual structure.
    # common fields: 'emotion' or 'emotions'[0].
    emotion = meta.get('emotion') or (meta.get('emotions') or ['neutral'])[0]
    emotion_to_images[emotion.lower()].append(filename)

print("Original sample size for each category:")
for emo, cnt in emotion_to_images.items():
    print(f"  {emo}: {len(cnt)}")

# Sampling parameters: Maximum 500 images per category, minimum 200 images per category
MAX_PER_CLASS = 500
MIN_PER_CLASS = 200

subset_dir = './findingemo_subset_images'
os.makedirs(subset_dir, exist_ok=True)

label_json = {}
selected_count = 0

for emotion, images in emotion_to_images.items():
    if len(images) <= MIN_PER_CLASS:
        selected_images = images
    else:
        selected_images = random.sample(images, min(MAX_PER_CLASS, len(images)))
    
    print(f"Sample {emotion}: {len(selected_images)}")
    
    for img_name in selected_images:
        src = os.path.join('./findingemo_full_images', img_name)
        dst = os.path.join(subset_dir, img_name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
        
        # CLIP requires a text description (to improve zero-shot performance).
        label_json[img_name] = f"a complex social scene evoking {emotion}"

    selected_count += len(selected_images)

print(f"Subset preparation complete! Total number of images: {selected_count}")
print("Image saved to:", subset_dir)
with open('findingemo_subset_labels.json', 'w', encoding='utf-8') as f:
    json.dump(label_json, f, indent=4, ensure_ascii=False)
print("Save the tag file to: findingemo_subset_labels.json")