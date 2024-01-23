import torch
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
train_data = scipy.io.loadmat('data/nist36_train.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 100, 1024, 64, 36
device = torch.device('cpu')
x = torch.from_numpy(np.asarray(train_x)).float()
y = torch.from_numpy(np.where(train_y == 1)[1])
# Create random Tensors to hold inputs and outputs.
#x1 = torch.randn(N, D_in)# requires_grad=True)
#y = torch.empty(N, dtype=torch.long).random_(D_out)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          #torch.nn.ReLU(),
          torch.nn.Sigmoid(),
          torch.nn.Linear(H, D_out),
          torch.nn.Softmax(),
        ).to(device)
loss_fn = torch.nn.CrossEntropyLoss()

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epoch = 10000
total_loss = []
total_accuracy = []
for t in range(epoch):
  # Forward pass: compute predicted y by passing x to the model.
  y_pred = model(x)
  # Compute and print loss.
  loss = loss_fn(y_pred, y)
  predict = y_pred.max(1)[1]
  A = predict== y
  acc = torch.numel(A[A == 1])/ len(A)
  total_accuracy.append(acc)
  if (t%100 == 0):
      print('epoch=',t,' ','loss=',loss.item(),' ','accuracy=',acc)
  total_loss.append(loss.item())
  optimizer.zero_grad()
  #model.zero_grad()
  loss.backward()
  # Calling the step function on an Optimizer makes an update to its parameters
  optimizer.step()

epoches = np.linspace(1.0,float(epoch),num = epoch)
plt.figure()
plt.plot(epoches,total_loss)
plt.title('loss')
plt.show()
plt.figure()
plt.plot(epoches,total_accuracy)
plt.title('accuracy')
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