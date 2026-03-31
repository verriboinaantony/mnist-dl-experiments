# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import BasicNet
from config import BATCH_SIZE, LEARNING_RATE, EPOCHS

# ── 1. DATA ───────────────────────────────────────────────
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(
    root='data',
    train=True,
    transform=transform,
    download=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ── 2. MODEL ──────────────────────────────────────────────
model = BasicNet()

# ── 3. LOSS AND OPTIMIZER ─────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# ── 4. TRAINING LOOP ──────────────────────────────────────
for epoch in range(EPOCHS):
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {loss.item():.4f}")


torch.save(model.state_dict(), 'basic_net.pth')
print("Model saved.")