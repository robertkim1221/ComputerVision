import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import string

train_data = scipy.io.loadmat('data/nist36_train.mat')
valid_data = scipy.io.loadmat('data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
initialize_weights(1024,hidden_size,params,'layer1')
initialize_weights(hidden_size,hidden_size,params,'hidden')
initialize_weights(hidden_size,hidden_size,params,'hidden2')
initialize_weights(hidden_size,1024,params,'output')
params['M_layer1'] , params['M_output'] = 0 , 0

# should look like your previous training loops
losses = []
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        #forward pass
        h1 = forward(xb,params,'layer1',relu)
        h2 = forward(h1,params,'hidden',relu)
        h3 = forward(h2,params,'hidden2',relu)
        probs = forward(h3,params,'output',sigmoid)
        # loss
        #loss, acc = compute_loss_and_acc(xb, probs)
        loss = (probs - xb)**2 #loss = sum(error of output to image)^2

        total_loss += loss.sum()
        # backward
        delta1 = 2*(probs - xb)
        delta2 = backwards(delta1,params,'output')
        delta3 = backwards(delta2,params,'hidden2',relu_deriv)
        delta4 = backwards(delta3,params,'hidden',relu_deriv)
        backwards(delta4,params,'layer1',relu_deriv)

        # apply gradient
        # Layer names
        layers = ['layer1', 'hidden', 'hidden2', 'output']

        # Apply gradient update with momentum for each layer
        for layer in layers:
            # Update momentum term for weights
            params[f'M_{layer}'] = 0.9 * params[f'M_{layer}'] - learning_rate * params[f'grad_W{layer}']
            # Update weights
            params[f'W{layer}'] += params[f'M_{layer}']
            # Update biases, note that biases typically don't have momentum terms
            if layer != 'output':  # Assuming that output layer does not have a bias term
                params[f'b{layer}'] -= learning_rate * params[f'grad_b{layer}']

    losses.append(total_loss/train_x.shape[0])
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

# plot loss curve
plt.plot(range(len(losses)), losses)
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(losses)-1)
plt.ylim(0, None)
plt.grid()
plt.show()

        
# Q5.3.1
# choose 5 labels (change if you want)
visualize_labels = ["A", "B", "C", "1", "2"]


# get 2 validation images from each label to visualize
visualize_x = np.zeros((2*len(visualize_labels), valid_x.shape[1]))
for i, label in enumerate(visualize_labels):
    idx = 26+int(label) if label.isnumeric() else string.ascii_lowercase.index(label.lower())
    choices = np.random.choice(np.arange(100*idx, 100*(idx+1)), 2, replace=False)
    visualize_x[2*i:2*i+2] = valid_x[choices]

# run visualize_x through your network
# name the output reconstructed_x
##########################
##### your code here #####
##########################
h1 = forward(visualize_x,params,'layer1',relu)
h2 = forward(h1,params,'hidden',relu)
h3 = forward(h2,params,'hidden2',relu)
reconstructed_x = forward(h3,params,'output',sigmoid)

# visualize
fig = plt.figure()
plt.axis("off")
grid = ImageGrid(fig, 111, nrows_ncols=(len(visualize_labels), 4), axes_pad=0.05)
for i, ax in enumerate(grid):
    if i % 2 == 0:
        ax.imshow(visualize_x[i//2].reshape((32, 32)).T, cmap="Greys")
    else:
        ax.imshow(reconstructed_x[i//2].reshape((32, 32)).T, cmap="Greys")
    ax.set_axis_off()
plt.show()


# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
h1 = forward(valid_x,params,'layer1',relu)
h2 = forward(h1,params,'hidden',relu)
h3 = forward(h2,params,'hidden2',relu)
val_out = forward(h3,params,'output',sigmoid)
total_psnr = []
for i in range(valid_x.shape[0]):
    psnr_noisy = peak_signal_noise_ratio(valid_x[i,:],val_out[i,:])
    total_psnr.append(psnr_noisy)
psnr = np.array(total_psnr).mean()
print(psnr)