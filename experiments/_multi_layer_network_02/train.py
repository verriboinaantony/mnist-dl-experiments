# train.py

import os
import sys

# ── find root folder dynamically ──────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


print("ROOT is:", ROOT)
print("Looking for config in:", ROOT)


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import BasicNet
from config import BATCH_SIZE, LEARNING_RATE, EPOCHS

# ── paths ─────────────────────────────────────────────────
DATA_DIR  = os.path.join(ROOT, 'data')
MODEL_DIR = os.path.join(ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── data ──────────────────────────────────────────────────
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    transform=transform,
    download=True
)

val_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ── model ─────────────────────────────────────────────────
model     = BasicNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# ── evaluate function ─────────────────────────────────────
def evaluate(loader):
    model.eval()
    correct    = 0
    total_loss = 0

    with torch.no_grad():
        for images, labels in loader:
            output      = model(images)
            loss        = criterion(output, labels)
            total_loss += loss.item()
            predictions = output.argmax(dim=1)
            correct    += (predictions == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset) * 100
    return avg_loss, accuracy

# ── training loop ─────────────────────────────────────────
print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7}")
print("-" * 55)

for epoch in range(EPOCHS):

    # training
    model.train()
    total_train_loss = 0
    correct_train    = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss   = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        predictions       = output.argmax(dim=1)
        correct_train    += (predictions == labels).sum().item()

    train_loss = total_train_loss / len(train_loader)
    train_acc  = correct_train / len(train_dataset) * 100

    # validation
    val_loss, val_acc = evaluate(val_loader)

    print(f"{epoch+1:>6} | {train_loss:>10.4f} | {train_acc:>8.2f}% | {val_loss:>8.4f} | {val_acc:>6.2f}%")

# ── save model ────────────────────────────────────────────
save_path = os.path.join(MODEL_DIR, 'multi_layer.pth')
torch.save(model.state_dict(), save_path)
print(f"\nModel saved to {save_path}")