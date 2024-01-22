import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

# Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, compositeH
# Q2.2.4

def load_match(img1, img2, opts):
    #find matches between two images
    matches, locs1, locs2 = matchPics(img1, img2, opts)
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

    return bestH2to1


def trim(img):
    #given an image, trim the black edges
    #find the first column (from left to right)that has a non-black pixel
    #trim the image from the left to that column
    #no need for the right side since the image is already trimmed

    for i in range(img.shape[1]):
        if np.any(img[:,i] != 0):
            img = img[:,i:]
            break
    return img


if __name__ == "__main__":
    left = cv2.imread('data/pittsburgh_left.jpg')
    right = cv2.imread('data/pittsburgh_right.jpg')
    opts = get_opts()

    #resize resultant image width to be twice the width of the original images
    #make new image of size (height, width*2, 3)
    height = left.shape[0]
    width = left.shape[1]
    blank_template = np.zeros((height,2*width,3), np.uint8)

    #place right image in the the right half of the new image
    blank_template[0:height, width:2*width] = right

    #find homography of left to right image
    H = load_match(left, blank_template, opts)

    #use compositeH to combine to images
    composite_img = compositeH(H, left, blank_template)

    #trim the black edges
    panorama = trim(composite_img)
    print(np.shape(panorama))
    panorama = np.array(255/np.max(panorama)*panorama, np.uint8)
    cv2.imwrite('result/panorama.jpg', panorama)
    cv2.imshow('PANORAMA', panorama)
    cv2.waitKey(0)


