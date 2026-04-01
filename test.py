# test.py

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import BATCH_SIZE

# ── tell python where to find the model ───────────────────
EXPERIMENT = 'experiments/01_basic_network'
sys.path.insert(0, os.path.join(ROOT, EXPERIMENT))
from model import BasicNet

# ── paths ─────────────────────────────────────────────────
DATA_DIR  = os.path.join(ROOT, 'data')
MODEL_PATH = os.path.join(ROOT, 'models', 'basic_net.pth')

# ── data ──────────────────────────────────────────────────
transform = transforms.Compose([transforms.ToTensor()])

test_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    transform=transform,
    download=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ── load model ────────────────────────────────────────────
model = BasicNet()
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

criterion = nn.CrossEntropyLoss()

# ── test ──────────────────────────────────────────────────
correct    = 0
total_loss = 0

with torch.no_grad():
    for images, labels in test_loader:
        output      = model(images)
        loss        = criterion(output, labels)
        total_loss += loss.item()
        predictions = output.argmax(dim=1)
        correct    += (predictions == labels).sum().item()

avg_loss = total_loss / len(test_loader)
accuracy = correct / len(test_dataset) * 100

print(f"Test Loss     : {avg_loss:.4f}")
print(f"Test Accuracy : {accuracy:.2f}%")