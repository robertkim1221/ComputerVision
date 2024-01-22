import numpy as np
import matplotlib.pyplot as plt

import os

from helper import visualize_keypoints, plot_3d_keypoint, connections_3d, colors, toHomogenous
from q3_2_triangulate import triangulate

# Insert your package here

"""
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.

Modified by Vineet Tambe, 2023.
"""


def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres=100):
    # assert pts1.shape[0] == pts2.shape[0]

    Ps = list()
    err = 0
    new_pts1 = []
    new_pts2 = []
    new_pts3 = []

    N = pts1.shape[0]
    for k in range(N):
        if pts1[k, 2] > Thres:
            new_pts1.append(pts1[k, :])
        if pts2[k, 2] > Thres:
            new_pts2.append(pts2[k, :])
        if pts2[k, 2] > Thres:
            new_pts3.append(pts3[k, :])

    new_pts1 = np.asarray(new_pts1)
    new_pts2 = np.asarray(new_pts2)
    new_pts3 = np.asarray(new_pts3)

    for i in range(N):
        x1, y1, __ = new_pts1[i, :]
        x2, y2, __ = new_pts2[i, :]
        x3, y3, __ = new_pts3[i, :]
        
        # construct A
        A0 =  y1*C1[2, :] - C1[1, :]
        A1 = C1[0, :] - x1*C1[2, :]
        A2 =  y2*C2[2, :] - C2[1, :]
        A3 = C2[0, :] - x2*C2[2, :]
        A4 =  y3*C3[2, :] - C3[1, :]
        A5 = C3[0, :] - x3*C3[2, :]
        A = np.stack((A0, A1, A2, A3, A4, A5), axis=0)

        # solve w, just find the null space
        U, s, Vt = np.linalg.svd(A)
        w_raw = Vt[-1, :] #(4,)
        w_3d = w_raw[0:3] / w_raw[3] #(3,)
        Ps.append(w_3d)
        
        # get reproject error
        w_homo = np.zeros((4, 1), dtype=np.float32)
        w_homo[0:3, 0] = w_3d
        w_homo[3, 0] = 1
        p1_rep = C1 @ w_homo
        p2_rep = C2 @ w_homo
        p3_rep = C3 @ w_homo

        x1_rep, y1_rep = p1_rep[0:2, 0] / p1_rep[2, 0]
        x2_rep, y2_rep = p2_rep[0:2, 0] / p2_rep[2, 0]
        x3_rep, y3_rep = p3_rep[0:2, 0] / p3_rep[2, 0]

        err += (x1_rep-x1)**2 + (y1_rep-y1)**2 + (x2_rep-x2)**2 + (y2_rep-y2)**2 + (x3_rep-x3)**2 + (y3_rep-y3)**2
        
    P = np.stack(Ps, axis=0)
    return P, err

"""
Q6.2 Plot Spatio-temporal (3D) keypoints
    :param car_points: np.array points * 3
"""

def plot_3d_keypoint_video(pts_3d_video):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(pts_3d_video.shape[0]):
        pts_3d = pts_3d_video[i]

        for j in range(len(connections_3d)):
            index0, index1 = connections_3d[j]
            xline = [pts_3d[index0, 0], pts_3d[index1, 0]]
            yline = [pts_3d[index0, 1], pts_3d[index1, 1]]
            zline = [pts_3d[index0, 2], pts_3d[index1, 2]]
            ax.plot(xline, yline, zline, color=colors[j])
    np.set_printoptions(threshold=1e6, suppress=True)
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    plt.show()

if __name__ == "__main__":
    pts_3d_video = []
    for loop in range(10):
        print(f"processing time frame - {loop}")

        data_path = os.path.join("data/q6/", "time" + str(loop) + ".npz")
        image1_path = os.path.join("data/q6/", "cam1_time" + str(loop) + ".jpg")
        image2_path = os.path.join("data/q6/", "cam2_time" + str(loop) + ".jpg")
        image3_path = os.path.join("data/q6/", "cam3_time" + str(loop) + ".jpg")

        im1 = plt.imread(image1_path)
        im2 = plt.imread(image2_path)
        im3 = plt.imread(image3_path)

        data = np.load(data_path)
        pts1 = data["pts1"]
        pts2 = data["pts2"]
        pts3 = data["pts3"]

        K1 = data["K1"]
        K2 = data["K2"]
        K3 = data["K3"]

        M1 = data["M1"]
        M2 = data["M2"]
        M3 = data["M3"]

        # Note - Press 'Escape' key to exit img preview and loop further
        #img = visualize_keypoints(im2, pts2)

        # TODO: YOUR CODE HERE
        C1 = K1 @ M1
        C2 = K2 @ M2
        C3 = K3 @ M3        
        P, err = MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres=80)
        
        pts_3d_video.append(P)
    
    print(err)
    pts_3d_video = np.asarray(pts_3d_video)

    # plotting 3d keypoints for some frames
    plot_3d_keypoint(pts_3d_video[0])
    plot_3d_keypoint(pts_3d_video[4])
    plot_3d_keypoint(pts_3d_video[8])

    plot_3d_keypoint_video(pts_3d_video)
