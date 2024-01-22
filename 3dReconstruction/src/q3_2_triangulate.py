import numpy as np
import matplotlib.pyplot as plt

from helper import camera2
from q2_1_eightpoint import eightpoint
from q3_1_essential_matrix import essentialMatrix

# Insert your package here


"""
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.

    Hints:
    (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
    (2) Solve for the least square solution using np.linalg.svd
    (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from 
        homogeneous coordinates to non-homogeneous ones)
    (4) Keep track of the 3D points and projection error, and continue to next point 
    (5) You do not need to follow the exact procedure above. 
"""


def triangulate(C1, pts1, C2, pts2):
    #initialize array for P and error
    Ps = list()
    err = 0

    #loop through all points
    N = pts1.shape[0]
    for i in range(N):
        x1, y1 = pts1[i, :]
        x2, y2 = pts2[i, :]
        
        # Compute A using derived formula
        A0 =  y1*C1[2, :] - C1[1, :]
        A1 = C1[0, :] - x1*C1[2, :]
        A2 =  y2*C2[2, :] - C2[1, :]
        A3 = C2[0, :] - x2*C2[2, :]
        A = np.stack((A0, A1, A2, A3), axis=0)

        # solve for lstsq solution using SVD
        __, __, Vt = np.linalg.svd(A)
    
        w_raw = Vt[-1, :]
        w_3d = w_raw[0:3] / w_raw[3] #normalize by last element
        Ps.append(w_3d)
        
        # calculate reprojection error using homogenous coordinates
        w_homo = np.zeros((4, 1), dtype=np.float32)
        w_homo[0:3, 0] = w_3d
        w_homo[3, 0] = 1

        p1_rep = C1 @ w_homo
        p2_rep = C2 @ w_homo

        # normalize reprojected points
        x1_rep, y1_rep = p1_rep[0:2, 0] / p1_rep[2, 0]
        x2_rep, y2_rep = p2_rep[0:2, 0] / p2_rep[2, 0]

        # calculate error and add each iteration
        err += (x1_rep-x1)**2 + (y1_rep-y1)**2 + (x2_rep-x2)**2 + (y2_rep-y2)**2
        
    P = np.stack(Ps, axis=0)
    return P, err

def findM2(F, pts1, pts2, intrinsics, filename="q3_3.npz"):
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']
    
    # CALCULATE E
    E = essentialMatrix(F, K1, K2)

    # CALCULATE M1 and M2
    M1 = np.array([ [ 1,0,0,0 ],
                    [ 0,1,0,0 ],
                    [ 0,0,1,0 ]  ])

    M2_list = camera2(E)

    #  TRIANGULATION
    C1 = K1.dot(M1)

    P = np.zeros( (pts1.shape[0],3) )
    M2 = np.zeros( (3,4) )
    C2 = np.zeros( (3,4) )
    prev_err = np.inf
    for i in range(M2_list.shape[2]):
        M2 = M2_list[:, :, i]
        C2 = K2.dot(M2)
        P_i, err = triangulate(C1, pts1, C2, pts2)
        if ( err<prev_err and np.min(P_i[:, 2])>=0):
            P = P_i
            M2 = M2
            C2 = C2
            prev_err = err
            
    np.savez(filename, M2=M2, C2=C2, P=P)

    return M2, C2, P


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    M2, C2, P = findM2(F, pts1, pts2, intrinsics)

    # Simple Tests to verify your implementation:
    M1 = np.hstack((np.identity(3), np.zeros(3)[:, np.newaxis]))
    C1 = K1.dot(M1)
    C2 = K2.dot(M2)
    P_test, err = triangulate(C1, pts1, C2, pts2)
    assert err < 500
