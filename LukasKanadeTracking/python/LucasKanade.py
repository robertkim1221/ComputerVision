import numpy as np
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    # set up the threshold
    ################### TODO Implement Lucas Kanade ###################
    
    # Define p as the initial guess
    p = p0

    # Extract rectangle coordinates and size of It and rect
    x1, y1, x2, y2 = rect
    rows_It, cols_It = It.shape
    rows_rect, cols_rect = x2 - x1, y2 - y1

    #Jacobian
    jac = np.array([[1,0],[0,1]]) #due to only translation

    # Steps to compute gradient of It and create objects for interpolation of It1
    # 1. create x, y coordinate of template image
    y = np.arange(0, rows_It, 1)
    x = np.arange(0, cols_It, 1)     
    # 2. create x, y coordinate of ROI
    c = np.linspace(x1, x2, int(cols_rect))
    r = np.linspace(y1, y2, int(rows_rect))

    # 3. create RectBivariateSpline object to interpolate the gradient within ROI
    spline = RectBivariateSpline(y, x, It)
    It1_interp = RectBivariateSpline(y, x, It1)

    # For Ix, Iy, first compute the gradient of It1 w.r.t x and y (flipped in np.gradient)
    Iy, Ix = np.gradient(It1)
    
    Ix_interp = RectBivariateSpline(y, x, Ix)
    Iy_interp = RectBivariateSpline(y, x, Iy)

    # 4. Image gradient of template image ROI can be precomputed before Lucas-Kanade loop
    col, row = np.meshgrid(c, r) # Create coordinate grid for ROI for interpolation
    T = spline.ev(row, col)   # Gradient of template image ROI

    for __ in range(int(num_iters)):
        # warp image using translation motion model
        x1_w, y1_w, x2_w, y2_w = x1+p[0], y1+p[1], x2+p[0], y2+p[1]

        # Coordinate matrix for It1 and Ix, Iy
        cw = np.linspace(x1_w, x2_w, int(cols_rect))  # create coordinate grid for warped image
        rw = np.linspace(y1_w, y2_w, int(rows_rect))

        col_warped, row_warped = np.meshgrid(cw, rw) # Note than new ROI changed due to p

        # Using the new ROI to interpolate the gradient of It1
        warpImg = It1_interp.ev(row_warped, col_warped) # Gradient for error calculation
        Ix_w = Ix_interp.ev(row_warped, col_warped) #Gradient for steepest descent
        Iy_w = Iy_interp.ev(row_warped, col_warped)        
        
        #compute error image
        err = T - warpImg
        errImg = err.reshape(-1,1) #Create column vector from the error matrix
        
        #gI is Nx2 matrix
        gI = np.vstack((Ix_w.ravel(),Iy_w.ravel())).T
        
        # Steepest descent (Nx2)
        A = gI @ jac 

        #compute dp
        dp = np.linalg.lstsq(A, errImg, rcond=1)[0]
        
        #update parameters
        p[0] += dp[0,0]
        p[1] += dp[1,0]

        #check if dp is smaller than threshold
        if np.linalg.norm(dp)**2 < threshold:
            break
    return p