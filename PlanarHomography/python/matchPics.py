import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

# Q2.1.4

def matchPics(I1, I2, opts):

        ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
        sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'

        # TODO: Convert Images to GrayScale
        I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
        I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
        
        # TODO: Detect Features in Both Images
        locs1 = corner_detection(I1, sigma)
        locs2 = corner_detection(I2, sigma)
        
        # TODO: Obtain descriptors for the computed feature locations
        desc1, locs1 = computeBrief(I1, locs1)
        decs2, locs2 = computeBrief(I2, locs2)       

        # TODO: Match features using the descriptors
        matches = briefMatch(desc1, decs2, ratio)

        return matches, locs1, locs2