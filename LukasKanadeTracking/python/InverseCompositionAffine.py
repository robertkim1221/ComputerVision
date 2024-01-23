import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import cv2
def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
    M0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    # Use homogeneous coordinates for simplicity
    M = np.vstack((M0, np.array([0.0, 0.0, 1.0])))

    # Make temporary M to update M in iterations
    M_temp = M0.flatten()
    
    # Precompute the template gradient
    Iy, Ix = np.gradient(It)
    Ix = Ix.flatten()
    Iy = Iy.flatten()
    
    # Create object for It1 interpolation (Works better than affine_transform)
    It1_interp=RectBivariateSpline(np.arange(It1.shape[0]),
                                np.arange(It1.shape[1]),
                                It1)
    
    # Define the template coordinates for warping within loop
    x1, y1, x2, y2 = 0, 0, It.shape[1], It.shape[0]
    x = np.arange(x1, x2)
    y = np.arange(y1, y2)
    X, Y = np.meshgrid(x,y)

    # Compute the steepest descent images
    A = np.array([Ix*X.flatten(), Ix*Y.flatten(), Ix, Iy*X.flatten(), Iy*Y.flatten(), Iy]).T
    for __ in range(int(num_iters)):

        # Warp the image using the given warp formula (Mx = x')
        X_w = M_temp[0] * X + M_temp[1] * Y + M_temp[2] # New coordinates of warped image
        Y_w = M_temp[3] * X + M_temp[4] * Y + M_temp[5] # is given by warpX and warpY 
        
        # Keep the warp within the original image
        # By simply use np.where to give boolean mask
        # where only warpX and warpY within the template are True
        mask = np.where((X_w >= x1) & (X_w <= x2) & 
                             (Y_w >= y1) & (Y_w <= y2), True, False)
        
        # Update the warpX and warpY to be within the template
        X_w, Y_w = X_w[mask], Y_w[mask]

        # Warp It1
        It1_warp = It1_interp.ev(Y_w, X_w)

        # Compute error
        error = (It1_warp - It[mask]).flatten()

        # Update steepest descent images
        A = A[mask.flatten()]

        # Compute dp
        dp = (np.linalg.inv(A.T @ A)) @ (A.T @ error)

        #update dM         
        M = np.vstack((M_temp.reshape(2,3), M[2,0:3])) # 3x3 from M_temp 
        dM = np.vstack((dp.reshape(2,3), M[2,0:3])) # 3x3 from dp

        # Add 1 to first two diagonal elements as given in affine warp function
        dM[0,0] += 1
        dM[1,1] += 1
        
        # Compute M
        M = M @ np.linalg.inv(dM)
        
        M_temp = M[0:2, 0:3].flatten() # update p

        if np.linalg.norm(dp)**2 < threshold:
            break
        
    # Return 2x3 matrix
    M = M[0:2, 0:3]
    return M