# test.py

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from experiments._basic_network_01 import BasicNet
from config import BATCH_SIZE

# ── DATA ──────────────────────────────────────────────────
transform = transforms.Compose([transforms.ToTensor()])

test_dataset = datasets.MNIST(
    root='data',
    train=False,
    transform=transform,
    download=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ── MODEL ─────────────────────────────────────────────────
model = BasicNet()
model.load_state_dict(torch.load('basic_net.pth', weights_only=True))
model.eval()

# ── LOSS ──────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()

# ── TESTING ───────────────────────────────────────────────
correct = 0
total_loss = 0

with torch.no_grad():
    for images, labels in test_loader:
        output = model(images)
        loss = criterion(output, labels)
        total_loss += loss.item()
        predictions = output.argmax(dim=1)
        correct += (predictions == labels).sum().item()

avg_loss = total_loss / len(test_loader)
accuracy = correct / len(test_dataset) * 100

print(f"Test Loss: {avg_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")