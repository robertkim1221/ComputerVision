import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None

    # padding parameter
    padding = 10
    # De-noise first 
    #image = skimage.restoration.denoise_bilateral(image, channel_axis=-1)
    # convert to grayscale
    gray = skimage.color.rgb2gray(image)
    # apply threshold
    thresh = skimage.filters.threshold_otsu(gray)
    bw = skimage.morphology.closing(gray < thresh,skimage.morphology.square(6))

    # dilation
    selem = skimage.morphology.square(5)
    bw = skimage.morphology.dilation(bw, selem)
#    bw = skimage.morphology.dilation(bw)
    # remove artifacts connected to image border
    cleared = skimage.segmentation.clear_border(bw)
    # label image regions
    label = skimage.measure.label(cleared)
    bw = np.invert(bw).astype(int)

    # loop through each region and append bounding box to bboxes
    for region in skimage.measure.regionprops(label):
        if region.area >= 500:      # set region threshold to be 300
            # Get the bounding box of the region
            minr, minc, maxr, maxc = region.bbox

            # Apply padding to each side of the bounding box
            minr = max(minr - padding, 0)
            minc = max(minc - padding, 0)
            maxr = min(maxr + padding, image.shape[0])
            maxc = min(maxc + padding, image.shape[1])

            # Create a larger bounding box with padding
            padded_box = [minr, minc, maxr, maxc]
            bboxes.append(padded_box)

    return bboxes, bw

