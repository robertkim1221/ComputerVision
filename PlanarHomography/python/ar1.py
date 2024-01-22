import cv2
import numpy as np
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
from helper import loadVid

def warpImage(opts):
    cover = cv2.imread('data/cv_cover.jpg')
    video = loadVid('data/book.mov')  # size is 511, 360, 640, 3 -> (num_frames, height, width, channels)
    source = loadVid('data/ar_source.mov')

    #need to scale potter image to be the same size as cover image
    potter = cv2.resize(source[0], (cover.shape[1], cover.shape[0]))


    #find matches between two images
    matches, locs1, locs2 = matchPics(cover, video[0], opts)
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
    composite_img = compositeH(bestH2to1, source[0], video[0])

    cv2.imshow('composite', composite_img)
    cv2.waitKey(0)
    pass



if __name__ == "__main__":

    opts = get_opts()
    warpImage(opts)


    #inlier_tol of 1.5 works better than 2.0
    


