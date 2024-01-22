import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

# Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
# Q2.2.4

def warpImage(opts):
    cover = cv2.imread('data/cv_cover.jpg')
    desk = cv2.imread('data/cv_desk.png')
    potter = cv2.imread('data/hp_cover.jpg')

    #need to scale potter image to be the same size as cover image
    potter = cv2.resize(potter, (cover.shape[1], cover.shape[0]))

    #find matches between two images
    matches, locs1, locs2 = matchPics(cover, desk, opts)
    #locs are in (y,x) format
    y1 = locs1[:,0]
    x1 = locs1[:,1]
    y2 = locs2[:,0]
    x2 = locs2[:,1]

    locs_x1 = np.array([x1, y1])
    locs_x1 = np.transpose(locs_x1)
    locs_x2 = np.array([x2, y2])
    locs_x2 = np.transpose(locs_x2)
    bestH2to1, inliers = computeH_ransac(locs_x1[matches[:,0]], locs_x2[matches[:, 1]], opts)

    print(bestH2to1)

    #composite image
    composite_img = compositeH(bestH2to1, potter, desk)

    cv2.imshow('composite', composite_img)
    cv2.waitKey(0)
    pass



if __name__ == "__main__":

    opts = get_opts()
    warpImage(opts)
    


