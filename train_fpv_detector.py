import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_search_dir = os.path.join("data", "fpv_frames")
model_save_path = "fpv_classifier.pth"

class CricketFPVDataset(Dataset):
    def __init__(self, search_dir, transform=None):
        self.transform = transform
        self.samples = []
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for root, _, files in os.walk(search_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_extensions:
                    full_path = os.path.join(root, file)
                    lower_name = full_path.lower()
                    
                    # 0 = Non-FPV (crowd, replays, side view), 1 = FPV (pitch delivery view)
                    if "non" in lower_name or "negative" in lower_name or "bg" in lower_name:
                        label = 0
                    elif "fpv" in lower_name or "positive" in lower_name or "pitch" in lower_name or "ball" in lower_name:
                        label = 1
                    else:
                        label = 1
                    self.samples.append((full_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = CricketFPVDataset(root_search_dir, transform=data_transforms)
print(f"Total valid image samples located: {len(dataset)}")

if len(dataset) == 0:
    raise RuntimeError(f"No image files (.jpg, .png, etc.) found under '{root_search_dir}'. Check if files need unzipping.")

# Train / Validation Split (80% / 20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Lightweight MobileNetV2 Backbone
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print(f"Training FPV Classifier on {device}...")
epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = (correct / total) * 100 if total > 0 else 0
    print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {running_loss/train_size:.4f} | Val Accuracy: {val_acc:.2f}%")

torch.save(model.state_dict(), model_save_path)
print(f"Trained model saved successfully as '{model_save_path}'.")