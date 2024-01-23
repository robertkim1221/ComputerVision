import numpy as np
from LucasKanadeAffine import *
from scipy.ndimage import affine_transform
from scipy.ndimage import binary_dilation, binary_erosion
from InverseCompositionAffine import *

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """

    # put your implementation here
    mask = np.zeros(image1.shape, dtype=bool)

    ################### TODO Implement Substract Dominent Motion ###################

    # Use either LKAffine or ICAffine to compute warp matrix M
    # M = LucasKanadeAffine(image1, image2, threshold, num_iters) # Uncomment for LKAffine
    M = InverseCompositionAffine(image1, image2, threshold, num_iters)

    # Warp image1 to image1_w using M
    image1_w = affine_transform(image1, M)

    # Compute difference between image1_w and image2
    diff = np.abs(image2-image1_w)
    mask = (diff > tolerance)

    # Use binary dilation first to fill in holes
    # Then use binary erosion to remove noise
    mask = binary_dilation(mask, iterations=3)
    mask = binary_erosion(mask, iterations=4)

    return mask.astype(bool)


