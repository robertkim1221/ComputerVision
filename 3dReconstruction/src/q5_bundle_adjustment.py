import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2, triangulate

import scipy.optimize

# Insert your package here
import random

# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""


def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:, 0], P_before[:, 1], P_before[:, 2], c="blue")
    ax.scatter(P_after[:, 0], P_after[:, 1], P_after[:, 2], c="red")
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


"""
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
"""

def ransacF(pts1, pts2, M, nIters=1000, tol=2):

    max_inliers = -np.inf

    for __ in range(nIters):
        ran_points = random.sample(range(0, pts1.shape[0]), 7)
        
        pts1_sample = pts1[ran_points]
        pts2_sample = pts2[ran_points]

        F_list = sevenpoint(pts1_sample, pts2_sample, M)
        
        for F_tmp in F_list:
            total_inliers = 0
            inlier_tmp = np.zeros(pts1.shape[0], dtype=bool)
            for k in range(pts1.shape[0]):
                #make homogenous points
                x1 = np.array(  [pts1[k,0], pts1[k,1], 1] ).reshape(1,3)
                x2 = np.array(  [pts2[k,0], pts2[k,1], 1] ).reshape(1,3)

                # use epipolar constraint to check if point is inlier
                if calc_epi_error(x1, x2, F_tmp) < tol:
                    total_inliers = total_inliers +1
                    inlier_tmp[k] = True
                else:
                    inlier_tmp[k] = False
                
            if total_inliers > max_inliers:
                max_inliers = total_inliers
                inliers = inlier_tmp
                F = F_tmp

    print("max inliers: ", max_inliers)

    return F, inliers

"""
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
"""

def rodrigues(r):
    zero = 1e-30 # threshold for checking if theta is close to 0
    theta = np.linalg.norm(r) # theta is the length of r
    
    if np.abs(theta) < zero:
        return np.eye(3, dtype=np.float32) # if ~0 then return identity matrix
    else:
        u = r / theta
        u_cross = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]], dtype=np.float32)
        u = u.reshape(3,1)

        # Rodrigues formula
        R = np.eye(3, dtype=np.float32) * np.cos(theta) + (1 - np.cos(theta)) * (u @ u.transpose()) + u_cross * np.sin(theta)
        
        return R


"""
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
"""


    

def invRodrigues(R):
        # Arctan as defined in pdf
    def arctan2(y, x):
        if isgreater(x, 0):
            return np.arctan(y / x)
        elif isgreater(0, x):
            return np.pi + np.arctan(y / x)
        elif isequal(x, 0) and isgreater(y, 0):
            return np.pi*0.5
        elif isequal(x, 0) and isgreater(0, y):
            return -np.pi*0.5
        
    def isequal(a,b): # to check if close to 0
        zero = 0.001
        return np.abs(a - b) < zero
    
    def isgreater(a,b): # to check if greater than 0
        zero = 0.001
        return a - b > zero
    
    def S_half(r): # function for half sphere
        length = np.sum(r**2)**0.5
        r1, r2, r3 = r[0], r[1], r[2]
        if (isequal(length, np.pi) and isequal(r1, r2) and isequal(r1, 0) and isgreater(0, r3)) or (isequal(r1, 0) and isgreater(0, r2)) or  isgreater(0, r1):
            return -r
        else:
            return r

    zero = 0.0001
    A = (R - R.transpose()) / 2
    a32, a13, a21 = A[2, 1], A[0, 2], A[1, 0]
    rho = np.array([[a32], [a13], [a21]], dtype=np.float32).T
    s = np.sum(rho**2)**0.5
    c = (R[0, 0]+R[1, 1]+R[2, 2] - 1) / 2.0
    if isequal(s, 0) and isequal(c, 1):
        return np.zeros((3, 1), dtype=np.float32)
    elif isequal(s, 0) and isequal(c, -1):
        V = R+np.eye(3, dtype=np.float32)
        # find a nonzero column of V
        mark = np.where(np.sum(V**2, axis=0) > zero)[0]
        v = V[:, mark[0]]
        u = v / (np.sum(v**2)**0.5)

        r = S_half(u*np.pi)
        return r
    elif not isequal(s, 0):
        u = rho / s
        theta = arctan2(s, c)
        return u*theta


"""
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
"""


def rodriguesResidual(K1, M1, p1, K2, p2, x):
    n = p1.shape[0]
    P = x[0:3*n].reshape(n, 3)
    r2 = x[3*n:3*n+3]
    t2 = x[3*n+3:3*n+6]

    R2 = rodrigues(r2)

    t2 = t2.reshape(3,1)
    M2 = np.concatenate((R2, t2), axis=1)

    P_h = np.concatenate( ( P, np.ones( (P.shape[0], 1) ) ), axis=1 ).transpose()
    
    p1_rep_h = K1 @ M1 @ P_h
    p1_rep = p1_rep_h[0:2, :] / p1_rep_h[2, :]
    p2_rep_h = K2 @ M2 @ P_h
    p2_rep = p2_rep_h[0:2, :] / p2_rep_h[2, :]

    p1_hat = p1_rep.transpose()
    p2_hat = p2_rep.transpose()
    e1 = (p1 - p1_hat).reshape(-1)
    e2 = (p2 - p2_hat).reshape(-1)

    residuals = np.concatenate((e1, e2), axis=0)

    return residuals


"""
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
"""


def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    obj_start = obj_end = 0

    R_init = M2_init[:, 0:3]
    r_init = invRodrigues(R_init)
    t_init = M2_init[:, 3].reshape([-1])

    # Ensure P_init is reshaped properly
    P_init_flattened = P_init.reshape(-1)

    # Construct the initial concatenated vector
    x_init  = np.hstack([P_init_flattened, r_init.ravel(), t_init])

    func = lambda x: (rodriguesResidual(K1, M1, p1, K2, p2, x)** 2).sum()
    res = scipy.optimize.minimize(func, x_init, options={'disp': True})

    x_new = res.x

    n = p1.shape[0]
    P_new = x_new[0:3*n].reshape(n, 3)
    r_new = x_new[3*n:3*n+3]
    t_new = x_new[3*n+3:3*n+6, None]

    R_new = rodrigues(r_new)

    # Construct the final optimized M2
    M2_new = np.hstack([R_new, t_new])

    # Objective function values
    obj_start = func(x_init)
    obj_end = func(x_new)

    return M2_new, P_new, obj_start, obj_end


if __name__ == "__main__":
    np.random.seed(1)  # Added for testing, can be commented out

    some_corresp_noisy = np.load(
        "data/some_corresp_noisy.npz"
    )  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    noisy_pts1, noisy_pts2 = some_corresp_noisy["pts1"], some_corresp_noisy["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))

    #displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(
        noisy_pts2
    )
    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2

    # Simple Tests to verify your implementation:
    from scipy.spatial.transform import Rotation as sRot

    rotVec = sRot.random()
    mat = rodrigues(rotVec.as_rotvec())

    assert np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3
    assert np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3

    # Visualization:
    np.random.seed(1)
    correspondence = np.load(
        "data/some_corresp_noisy.npz"
    )  # Loading noisy correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")
    M = np.max([*im1.shape, *im2.shape])

    # TODO: YOUR CODE HERE
    """
    Call the ransacF function to find the fundamental matrix
    Call the findM2 function to find the extrinsics of the second camera
    Call the bundleAdjustment function to optimize the extrinsics and 3D points
    Plot the 3D points before and after bundle adjustment using the plot_3D_dual function
    """
    # use the same output from ransacF and find points that are inliers
    pts1_inliers= pts1[np.where(inliers)]
    pts2_inliers= pts2[np.where(inliers)]

    # call findM2 to find extrinsics of second camera
    M2, C2, P = findM2(F, pts1_inliers, pts2_inliers, intrinsics, "q5_1.npz")

    # call bundleAdjustment to optimize extrinsics and 3D points
    M1 = np.array([ [ 1,0,0,0 ],
                    [ 0,1,0,0 ],
                    [ 0,0,1,0 ]  ])

    M2_new, P_new, obj_start, obj_end = bundleAdjustment(K1, M1, pts1_inliers, K2, M2, pts2_inliers, P)

    # reprojection error
    print('Reprojection error before bundle adjustment: ', obj_start)
    print('Reprojection error after bundle adjustment: ', obj_end)
    
    # plot 3D points before and after bundle adjustment
    plot_3D_dual(P, P_new)

    # print('F: ', F)