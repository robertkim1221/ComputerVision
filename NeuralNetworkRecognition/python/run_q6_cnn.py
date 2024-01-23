import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io
import numpy as np

# Load your training data
train_data = scipy.io.loadmat('data/nist36_train.mat')
train_x, train_y = train_data['train_data'], train_data['train_labels']

# Reshape your data to (batch_size, channels, height, width)
train_x = train_x.reshape(-1, 1, 32, 32) # The -1 infers the batch size

# Convert to PyTorch tensors
x = torch.from_numpy(train_x).float()
y = torch.from_numpy(np.where(train_y == 1)[1]).long()

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 8 * 8, 64) # The input dimension here should match the output from the last conv/pool layers
        self.fc2 = nn.Linear(64, 36) # Output layer with 36 classes
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # Apply convolution, ReLU, and pooling
        x = self.pool(F.relu(self.conv2(x))) # Apply convolution, ReLU, and pooling
        
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 32 * 8 * 8) # This should match the dimension expected by the first fully connected layer
        x = F.relu(self.fc1(x)) # Apply fully connected layer and ReLU
        x = self.fc2(x) # Apply output layer
        return x

# Instantiate the model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # use MPS (for macbook) if available
x = x.to(device) #move tensor to device
y = y.to(device)
model = CNN().to(device)

# Loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epoch = 500
total_loss = []
total_accuracy = []

for t in range(epoch):
    y_pred = model(x) # Forward pass
    loss = loss_fn(y_pred, y) # Compute loss
    predict = y_pred.max(1)[1]
    acc = (predict == y).float().mean() # Calculate accuracy
    total_accuracy.append(acc.item())
    
    print(f'epoch={t} loss={loss.item()} accuracy={acc.item()}')
    total_loss.append(loss.item())
    
    optimizer.zero_grad()
    loss.backward() # Backward pass
    optimizer.step() # Update weights

# Plotting
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

# Run the model on the test set and calculate the accuracy
test_data = scipy.io.loadmat('data/nist36_test.mat')
test_x, test_y = test_data['test_data'], test_data['test_labels']

# Reshape and convert the test data to tensors
test_x = test_x.reshape(-1, 1, 32, 32)  # The -1 infers the batch size
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(np.where(test_y == 1)[1]).long()

# Move the test data to the device
test_x = test_x.to(device)
test_y = test_y.to(device)

# Put the model in evaluation mode
model.eval()

# No need to compute gradients (for faster computation and less memory usage)
with torch.no_grad():
    # Get predictions from the model
    test_pred = model(test_x)
    # Convert predictions to class labels
    predicted_labels = test_pred.argmax(1)
    # Calculate the accuracy
    accuracy = (predicted_labels == test_y).float().mean().item()

print(f'Test Accuracy: {accuracy * 100:.2f}%')