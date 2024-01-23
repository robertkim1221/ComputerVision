import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
from LucasKanade import *

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=1e-2,
    help='dp threshold of Lucas-Kanade for terminating optimization',
)
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("data/carseq.npy")
rect = [59, 116, 145, 151]

# Create a list to store the resulting bounding boxes
rectList = []

# Run Lucas-Kanade Loop on every frame and save resulting bounding boxes
seq_len = seq.shape[2]
for i in range(seq_len):
    # Skip first frame
    if (i == 0):
        continue
    print("Processing frame %d" % i)

    # Make a deep copy and append to resultant list to avoid changes in list after rect is changed
    rect_cp = rect.copy()
    rectList.append(rect_cp)

    # Run Lucas-Kanade
    It = seq[:,:,i-1]
    It1 = seq[:,:,i]
    p = LucasKanade(It, It1, rect, threshold, num_iters)

    # Update rect for the next frame
    rect[0] += p[0]
    rect[1] += p[1]
    rect[2] += p[0]
    rect[3] += p[1]

    # Check resulting image with bounding box at every 100 frames
    if i % 100 == 0 or i == 1:
        plt.figure()
        plt.imshow(seq[:,:,i],cmap='gray')
        bbox = patches.Rectangle((int(rect[0]), int(rect[1])), rect[2] - rect[0], rect[3] - rect[1],
                                    fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(bbox)
        plt.title('frame %d'%i)
        plt.show()

np.save('carseqrects.npy',rectList)
print("Successfully saved carseqrects.npy")