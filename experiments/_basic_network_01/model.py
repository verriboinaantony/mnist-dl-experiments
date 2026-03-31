# model.py

import torch.nn as nn

class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# transform = transforms.Compose([transforms.ToTensor()])

# train_dataset = datasets.MNIST(
#     root='data',
#     train=True,
#     transform=transform,
#     download=True
# )

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=64,    # send 64 images at a time
#     shuffle=True      # mix the order every epoch
# )

# images, labels = next(iter(train_loader))

# model = nn.Sequential(
#     nn.Flatten(),           # converts [1, 28, 28] → [784]
#     nn.Linear(784, 256),    # layer 1
#     nn.ReLU(),              # activation
#     nn.Linear(256, 128),    # layer 2
#     nn.ReLU(),
#     nn.Linear(128, 10)      # output layer — 10 classes (digits 0-9)
# )

# output = model(images)
# print("Output shape:", output.shape)

# criterion = nn.CrossEntropyLoss()

# loss = criterion(output, labels)
# print("Loss:", loss.item())

# optimizer = optim.SGD(model.parameters(), lr=0.01)
# print("Optimizer ready")

# # 1. clear old gradients
# optimizer.zero_grad()

# # 2. pass images through model
# output = model(images)

# # 3. calculate loss
# loss = criterion(output, labels)

# # 4. calculate gradients
# loss.backward()

# # 5. update weights
# optimizer.step()

# # fresh forward pass with updated weights
# new_output = model(images)

# # recalculate loss with new weights
# new_loss = criterion(new_output, labels)

# print("Loss before step:", loss.item())
# print("Loss after step: ", new_loss.item())

# # print("Batch of images shape:", images.shape)
# # print("Batch of labels shape:", labels.shape)
# # print("First 10 labels:", labels[:10])

# # import matplotlib.pyplot as plt

# # # take the first image from the batch
# # single_image = images[0]  # shape is [1, 28, 28]

# # # squeeze removes the channel dimension for plotting
# # # shape becomes [28, 28]
# # plt.imshow(single_image.squeeze(), cmap='gray')
# # plt.title(f"Label: {labels[0].item()}")
# # plt.axis('off')
# # plt.show()