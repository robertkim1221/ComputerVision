import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import *

# write your script here, we recommend the above libraries for making your animation

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
parser.add_argument(
    '--template_threshold',
    type=float,
    default=5,
    help='threshold for determining whether to update template',
)
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load('data/carseq.npy')
rect = [59, 116, 145, 151]

# first frame is template for tracking
seq0 = seq[:,:,0]

# for plotting
rectList = np.load('carseqrects.npy')
width = rect[3] - rect[1]
length = rect[2] - rect[0]

# Initialize list of corrected rects
rectList_new = [rect]

# Loop through frames
seq_len = seq.shape[2]
for i in range(seq_len):
    # Skip first frame
    if i == 0:
        continue
    
    # Check process
    print("Processing frame %d" % i)

    # Lucas Kanade
    It = seq[:,:,i-1]
    It1 = seq[:,:,i]
    prev_rect = rectList_new[-1] # Use last updated rect
    p = LucasKanade(It, It1, prev_rect, threshold, num_iters)

    # Update rect but wait for template correction
    x1, y1, x2, y2 = prev_rect
    cur_rect = [x1 + p[0], y1 + p[1], x2 + p[0], y2 + p[1]]
    
    # Template Correction to mitigate template drift using first frame as template
    p_x, p_y = cur_rect[0] - rect[0], cur_rect[1] - rect[1]
    p0 = LucasKanade(seq0, It1, rect, threshold, num_iters, np.array((p_x, p_y)))
    
    # If the template correction is small, update rect
    if np.linalg.norm(p) <= template_threshold:
        # Update rect
        x1, y1, x2, y2 = rect
        cur_rect_refined = (x1+p0[0], y1+p0[1], x2+p0[0], y2+p0[1])
        # Update rectList_new
        rectList_new.append(cur_rect_refined)
    else:
        # Update rectList_new if template correction is large
        rectList_new.append(cur_rect)

    # Plotting
    if i % 100 == 0 or i == 1:
        plt.figure()
        plt.imshow(seq[:,:,i],cmap='gray')
        bbox0 = patches.Rectangle((int(rectList[i-1][0]), int(rectList[i-1][1])), length, width,
                                    fill=False, edgecolor='blue', linewidth=2)
        plt.gca().add_patch(bbox0)
        bbox1 = patches.Rectangle((int(rectList_new[i-1][0]), int(rectList_new[i-1][1])), length, width,
                                    fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(bbox1)
        plt.title('frame %d' % i)
        plt.show()

np.save('carseqrects-wcrt.npy',rectList_new)
print("Successfully saved carseqrects-wcrt.npy")