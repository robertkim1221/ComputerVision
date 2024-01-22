import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
from matplotlib import pyplot
import scipy.ndimage

#Q2.1.6

def rotTest(opts):

    # TODO: Read the image and convert to grayscale, if necessary
    image = cv2.imread('data/cv_cover.jpg')
    num_matches = []
    for i in range(36):

        # TODO: Rotate Image
        rot = scipy.ndimage.rotate(image, i*10)

        # TODO: Compute features, descriptors and Match features
        matches, locs1, locs2 = matchPics(image, rot, opts)
        # Append the number of matches to the list
        num_matches.append(len(matches))

    # TODO: Create a histogram
    angles = np.arange(0, 360, 10)
    pyplot.hist(angles, bins=angles, weights=num_matches, edgecolor='black')

    # TODO: Configure plot labels and title
    pyplot.xlabel('Rotation (degrees)')
    pyplot.ylabel('Matches')
    pyplot.title('Histogram of Matches vs. Rotation Angle')

    # TODO: Display histogram
    pyplot.show()


    pass 


    # TODO: Display histogram
    pyplot.show()

if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)
