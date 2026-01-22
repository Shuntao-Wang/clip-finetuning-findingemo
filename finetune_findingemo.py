# finetune_findingemo.py
import torch
import clip
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
from tqdm import tqdm

class ClipDataset(Dataset):
    def __init__(self, image_folder, label_json, preprocess):
        with open(label_json, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)
        self.image_paths = list(self.labels.keys())
        self.image_folder = image_folder
        self.preprocess = preprocess
        self.tokenize = clip.tokenize

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.image_folder, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.preprocess(image)
        except Exception as e:
            print(f"Image loading failed {img_path}: {e}")
            # Return a blank image to prevent crashes.
            image = torch.zeros(3, 224, 224)
        
        text = self.tokenize(self.labels[img_name])[0]
        return image, text

# ==================== Configuration parameters ====================
image_folder = './findingemo_subset_images'
dataset_json = './findingemo_subset_labels.json'
model_name = "ViT-L/14"
batch_size = 64          # The subset is small, so it can be increased appropriately.
epochs = 10
learning_rate = 1e-5
mixed_precision = True
save_dir = './models'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'ViT-L-14-findingemo.pt')
# ================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Equipment used: {device}")

model, preprocess = clip.load(model_name, device=device, jit=False)

dataset = ClipDataset(image_folder, dataset_json, preprocess)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
if mixed_precision:
    scaler = torch.cuda.amp.GradScaler()

model.train()
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    total_loss = 0
    for images, texts in tqdm(dataloader):
        images = images.to(device)
        texts = texts.to(device)

        optimizer.zero_grad()
        if mixed_precision:
            with torch.cuda.amp.autocast():
                logits_per_image, _ = model(images, texts)
                loss = torch.nn.functional.cross_entropy(logits_per_image, torch.arange(len(images)).to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits_per_image, _ = model(images, texts)
            loss = torch.nn.functional.cross_entropy(logits_per_image, torch.arange(len(images)).to(device))
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    print(f"Average loss: {total_loss / len(dataloader):.4f}")

print("Fine-tuning complete! Save the model...")
torch.save(model.state_dict(), save_path)
print(f"The model has been saved to: {save_path}")
print("It can be loaded directly in the future: clip.load('ViT-L-14-findingemo', device='cuda')")