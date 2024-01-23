import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

# Define transformations and normalization for CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) #mean and std of CIFAR-10
])

# Load CIFAR-10 data
train_dataset = datasets.CIFAR10(root='data/cifar10', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='data/cifar10', train=False, download=True, transform=transform)

# Data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the CNN model adjusted for CIFAR-10
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Adjust the conv1 to take 3-channel images
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        # Output layer adjusted for 10 classes
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model and move it to the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = CNN().to(device)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epoch = 100 # it takes too long to train, so only train for 30 epochs, which is good enough to get accuracy > 90%
total_loss = []
total_accuracy = []

for t in range(epoch):
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += labels.size(0)
        
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.float() / total_samples
    total_loss.append(epoch_loss)
    total_accuracy.append(epoch_acc)
    print(f'Epoch {t}/{epoch - 1} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}')

# Plotting
# Convert the accuracy and loss from tensors to NumPy arrays if they are not already
if isinstance(total_accuracy[0], torch.Tensor):
    total_accuracy = [x.cpu().item() for x in total_accuracy]
if isinstance(total_loss[0], torch.Tensor):
    total_loss = [x.cpu().item() for x in total_loss]

epoches = np.arange(1, epoch + 1)
plt.figure()
plt.plot(epoches, total_loss)
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.figure()
plt.plot(epoches, total_accuracy)
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Test the model
test_dataset = datasets.CIFAR10(root='data/cifar10', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

model.eval()  # Set model to evaluation mode
running_corrects = 0
total_samples = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += labels.size(0)

accuracy = running_corrects.float() / total_samples
print(f'Test Accuracy: {accuracy * 100:.2f}%')