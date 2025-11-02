import os
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

# --------------------------
# Paths and configuration
# --------------------------
DATA_DIR = os.path.join("data", "spectrograms")
os.makedirs(DATA_DIR, exist_ok=True)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# --------------------------
# Dataset preparation
# --------------------------
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

data_len = len(dataset)
train_len = int(data_len * 0.8)
test_len = data_len - train_len

train_set, test_set = random_split(dataset, [train_len, test_len])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

# --------------------------
# Model setup
# --------------------------
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

# Freeze base parameters
for p in model.parameters():
    p.requires_grad = False

# Replace the classification head for your binary task
num_classes = 2
model.heads = nn.Sequential(
    nn.Linear(in_features=768, out_features=num_classes),
    nn.Softmax(dim=1)
)

# --------------------------
# Loss and optimizer
# --------------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.heads.parameters(), lr=0.003, momentum=0.9)

# --------------------------
# Training loop
# --------------------------
EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for _, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f}")

# --------------------------
# Save full model (NOT state_dict)
# --------------------------
os.makedirs("models", exist_ok=True)
torch.save(model, "models/vit.pth")
print("✅ Full ViT model trained and saved as models/vit.pth")
