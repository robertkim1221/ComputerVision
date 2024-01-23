import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ################### TODO Implement Lucas Kanade Affine ###################

    # Initializers for p
    dp = np.zeros([6])

    # Create x,y coordinates for It
    rows, cols = It.shape
    yt = np.linspace(0, rows, rows).astype(int)
    xt = np.linspace(0, cols, cols).astype(int)
    y, x = np.meshgrid(yt,xt)
    y = y.flatten()
    x = x.flatten()

    # Gradient of It1 in x and y
    Iy, Ix = np.gradient(It1)

    # Iterations of Lucas Kanade Affine
    for __ in range(int(num_iters)):
        # Warp image using affine transform for each gradients
        It1_w = affine_transform(It1,M)
        Ix_w = affine_transform(Ix,M)
        Iy_w = affine_transform(Iy,M)

		# Update gradients Ix Iy
        Ix_w = Ix.flatten()
        Iy_w = Iy.flatten()

        # Use new x,y coordinates for to compute steepest descent
        rows1, cols1 = It1_w.shape
        yt1 = np.linspace(0, rows1, rows1).astype(int)
        xt1 = np.linspace(0, cols1, cols1).astype(int)
        y1, x1 = np.meshgrid(yt1,xt1)
        y1 = y1.flatten()
        x1 = x1.flatten()

		# Compute error
        err = It - It1_w # entire It is template
        errImg = err.flatten() # reshape() does not work here

        # Iterate through each pixel and compute A for each row (pixel)
        A = [] # Nx6
        for i in range(len(x)):
            # A = delI * jacobian
            # Since delI is Nx2, and jaxobian is 2x6, A is Nx6
            # del I = [[Ix1, Iy1], ... , [Ixn, Iyn]]
            #Jacobian dW/dp is given as [[x, y, 1, 0, 0, 0], [0, 0, 0, x, y, 1]]
            Ai = [x1[i]*Ix_w[i], y1[i]*Ix_w[i], Ix_w[i] ,x1[i]*Iy_w[i], y1[i]*Iy_w[i], Iy_w[i]]
            A.append(Ai)

		# Compute dp
        dp = np.linalg.lstsq(A, errImg, rcond=1)[0]
        
        # Update M
        M += [[ dp[0] , dp[1] , dp[2] ],[ dp[3] ,dp[4] , dp[5] ]]

        # Check if dp is smaller than threshold     
        if(np.linalg.norm(dp)<threshold):
            break

        return M