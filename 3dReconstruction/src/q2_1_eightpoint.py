import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF

# Insert your package here


"""
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
        (Remember to use the normalized points instead of the original points)
    (6) Unscale the fundamental matrix
"""

def eightpoint(pts1, pts2, M):
    # scale the given points by M
    # so that difference in magnitude of points is not too large -> better SVD
    T =  np.diag([ 1/M , 1/M , 1.0])

    # use matrix T to scale the points (use homogenous coordinates)
    pts1_s = T.dot(toHomogenous(pts1).T).T
    pts2_s = T.dot(toHomogenous(pts2).T).T

    # de-homogenize the points
    pts1_s = pts1_s[:, :2]
    pts2_s = pts2_s[:, :2]

    # compute eight point algorithm's equation
    AF = np.zeros((pts1_s.shape[0], 9))
    for i in range(pts1_s.shape[0]):
        # AF = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
        AF[i, :] = [ pts2_s[i,0]*pts1_s[i,0] , pts2_s[i,0]*pts1_s[i,1] , pts2_s[i,0], 
                    pts2_s[i,1]*pts1_s[i,0] , pts2_s[i,1]*pts1_s[i,1] , pts2_s[i,1], 
                    pts1_s[i,0], pts1_s[i,1], 1  ]

    # solve for F matrix using SVD
    __, __, vt = np.linalg.svd(AF)
    # f is the last column of v, so the last raw of vt
    f = vt[-1, :].reshape(3, 3)

    f = refineF(f, pts1_s, pts2_s)

    F = T.T.dot(f).dot(T)

    #scale F so that F[2,2] = 1
    F = F / F[2, 2]

    np.savez("q2_1.npz", F, M)

    return F


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    # correspondence = np.load("data/some_corresp_noisy.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    print(F)
    # Q2.1
    displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)
    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2
    assert np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1