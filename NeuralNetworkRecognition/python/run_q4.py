import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir("images"):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join("images", img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        plt.gca().add_patch(rect)
    plt.show()

    # find the rows using..RANSAC, counting, clustering, etc.
    # initialize row list
    row = []
    for i in range(len(bboxes)):
        minr, minc, maxr, maxc = bboxes[i]
        row.append(minr)

    #start with first row and move on to the next if the difference is greater than some threshold
    num_row = 0
    last_row = row[0]
    lines = []
    m = 0
    for j in range(len(row)):
        if row[j] - last_row > 100: # set threshold to be 100
            each_line = bboxes[m:j] # get the bounding boxes in each row
            num_row += 1            # move to next row
            lines.append(each_line) # append the bounding boxes in each row
            m = j                   # start from the next row
        if j == len(row)-1:         # for the last row
            lines.append(bboxes[m:j])
        last_row = row[j]           # update last row
    num_row = num_row+1             # number of rows is +1  

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string

    letters = np.array(
        [_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)]
    )
    params = pickle.load(open("q3_weights.pickle", "rb"))

    text=''
    for i in range(num_row):
        one_line = np.array(lines[i])                   # get the bounding boxes in each row
        one_line = one_line[one_line[:,1].argsort()]    # sort the bounding boxes in each row by x
        num_element = len(one_line)                     # number of elements in each row
        diff_col = np.diff(one_line[:,1])               # get the difference between x of each element
        threshold = np.mean(diff_col)                   # set threshold to be the mean of the difference
        # for each column in a row
        for j in range(num_element):
            # extract data within the bounding box
            pos = one_line[j,:]
            data = bw[int(pos[0]):int(pos[2]),int(pos[1]):int(pos[3])]
            size_c = pos[3] - pos[1]    #column and row size
            size_r = pos[2] - pos[0]
            
            #add padding to make it square
            if size_c > size_r:
                    pad_size = int(size_c/5)
                    patch_row_1 = (size_c + 2*pad_size - size_r) // 2
                    patch_row_2 = size_c + 2*pad_size - patch_row_1 - size_r
                    data = np.pad(data,
                                  [(patch_row_1,patch_row_2),(pad_size,pad_size)],
                                  mode = 'constant',
                                  constant_values=1)
            else:
                    pad_size = int(size_r/5)
                    patch_column_1 = (size_r + 2*pad_size - size_c) // 2
                    patch_column_2 = size_r + 2*pad_size - patch_column_1 - size_c
                    data = np.pad(data,
                                  [(pad_size,pad_size),(patch_column_1,patch_column_2)],
                                  mode = 'constant',
                                  constant_values=1)
                    
            data = skimage.transform.resize(data,(32,32)).T
            data = data < data.max()
            data = data==0
            data = data.reshape(1,1024)
            selem = skimage.morphology.square(2)
            data = skimage.morphology.dilation(data, selem)

            #predict the label using neural network
            h1 = forward(data,params,'layer1')
            probs = forward(h1,params,'output',softmax)
            predict_label = np.argmax(probs,axis = 1)
            predict_label = int(predict_label)

            if j != (num_element-1):
                pos_next = one_line[j+1,:]
                if np.abs(pos[1] - pos_next[1]) < 1.5 * threshold:#check for spaces
                    text = text + letters[predict_label]
                else:
                    text = text + letters[predict_label]
                    text = text + ' '#
            if j == (num_element-1):
                    text = text + letters[predict_label]
        text = text + '\n'

    print('image',img)
    print(text)