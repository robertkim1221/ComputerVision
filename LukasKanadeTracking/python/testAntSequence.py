import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from SubtractDominantMotion import *
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=1e-2,
    help='dp threshold of Lucas-Kanade for terminating optimization',
)
parser.add_argument(
    '--tolerance',
    type=float,
    default=0.2,
    help='binary threshold of intensity difference when computing the mask',
)
parser.add_argument(
    '--seq_file',
    default='data/antseq.npy',
)

args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance
seq_file = args.seq_file

seq = np.load(seq_file)

'''
HINT:
1. Create an empty array 'masks' to store the motion masks for each frame.
2. Set the initial mask for the first frame to False.
3. Use the SubtractDominantMotion function to compute the motion mask between consecutive frames.
4. Use the motion 'masks; array for visualization.
'''

frames = [seq[:, :, i] for i in range(seq.shape[2])]
for i in range(0, len(frames) - 1):
    frame, next_frame = frames[i], frames[i + 1]
    mask = SubtractDominantMotion(frame, next_frame, threshold, num_iters, tolerance)
    img = np.repeat(next_frame[:, :, np.newaxis], 3, axis=2)
    img[:, :, 2][mask == 1] = 1
    cv2.imshow('image', img)
    cv2.waitKey(1)
    if i in [29, 59, 89, 119]:
        cv2.imwrite("result/q2.3ant_{}.jpg".format(i), img * 255)

