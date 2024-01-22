import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from q2_1_eightpoint import eightpoint
from q3_2_triangulate import findM2
from q4_1_epipolar_correspondence import epipolarCorrespondence

# Insert your package here


"""
Q4.2: Finding the 3D position of given points based on epipolar correspondence and triangulation
    Input:  temple_pts1, chosen points from im1
            intrinsics, the intrinsics dictionary for calling epipolarCorrespondence
            F, the fundamental matrix
            im1, the first image
            im2, the second image
    Output: P (Nx3) the recovered 3D points
    
    Hints:
    (1) Use epipolarCorrespondence to find the corresponding point for [x1 y1] (find [x2, y2])
    (2) Now you have a set of corresponding points [x1, y1] and [x2, y2], you can compute the M2
        matrix and use triangulate to find the 3D points. 
    (3) Use the function findM2 to find the 3D points P (do not recalculate fundamental matrices)
    (4) As a reference, our solution's best error is around ~2200 on the 3D points. 

    Modified by Vineet Tambe, 2023.
"""


def compute3D_pts(temple_pts1, intrinsics, F, im1, im2):
    # Initialize template points
    x1 = temple_pts1[:,0].astype(int).flatten()
    y1 = temple_pts1[:,1].astype(int).flatten()

    M1 = np.array([ [ 1,0,0,0 ],
                    [ 0,1,0,0 ],
                    [ 0,0,1,0 ]  ])

    C1 = K1.dot(M1)

    # FIND EPIPOLAR PTS2 CORRESPONDANCES
    pts1_new = []
    pts2_new = []

    # for each point in temple_pts1, find the corresponding point in temple_pts2
    for i in range(x1.shape[0]):
        x2, y2 = epipolarCorrespondence(im1, im2, F, x1[i], y1[i])
        if x2 is not None:
            pts1_new.append([ x1[i], y1[i] ])
            pts2_new.append([ x2, y2 ])

    # convert to numpy array
    pts1_new = np.asarray(pts1_new)
    pts2_new = np.asarray(pts2_new)

    # Find 3D points using triangulation
    M2_best, C2_best, P = findM2(F, pts1_new, pts2_new, intrinsics)

    # Save to npz
    np.savez('q4_2.npz', F=F, M1=M1, M2=M2_best, C1=C1 , C2=C2_best )

    return P


def plot_3D(P):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2])
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


"""
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
"""

if __name__ == "__main__":
    temple_coords = np.load("data/templeCoords.npz")
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    # Call compute3D_pts to get the 3D points and visualize using matplotlib scatter
    temple_pts1 = np.hstack([temple_coords["x1"], temple_coords["y1"]])
    print(temple_pts1)
    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    P = compute3D_pts(temple_pts1, intrinsics, F, im1, im2)
    # Visualize
    fig = plt.figure()
    ax = Axes3D(fig)
    plot_3D(P)
