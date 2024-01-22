import numpy as np
import cv2


def computeH(x1, x2):
    #Q2.2.1
    # TODO: Compute the homography between two sets of points

    # Number of points
    N = len(x1)

    # Create the matrix A for the homography equation Ax = 0
    # 2*N because of x AND y and 9 because of 9 elements in h vector(so #of cols in A and # of rows in h are same)
    A = np.zeros((2 * N, 9))

    for i in range(N):
        x, y = x1[i, 0], x1[i, 1] #x and y of source
        xp, yp = x2[i, 0], x2[i, 1] #x and y of destination


        #From Q1.2.3, A_i is found and because of the 2*N, we have 2*A_i
        A[2 * i, :] = [-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp]
        A[2 * i + 1, :] = [0, 0, 0, -x, -y, -1, x * yp, y * yp, yp]

    # Solve the equation Ah = 0 using svd
    h = np.linalg.svd(A)[2][-1, :]
    
    # Reshape h into the matrix H
    H2to1 = h.reshape((3, 3))
    return H2to1


def computeH_norm(x1, x2):
    #Q2.2.2
    # TODO: Compute the centroid of the points
    x1_centroid = np.mean(x1, axis=0)
    x2_centroid = np.mean(x2, axis=0)

    # TODO: Shift the origin of the points to the centroid
    x1_shifted = x1 - x1_centroid
    x2_shifted = x2 - x2_centroid

    # TODO: Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    max_distance1 = np.max(np.linalg.norm(x1_shifted, axis=1))
    max_distance2 = np.max(np.linalg.norm(x2_shifted, axis=1))

    if max_distance1 == 0:
        norm_scale1 = 1
    else:
        norm_scale1 = np.sqrt(2) / max_distance1

    if max_distance2 == 0:
        norm_scale2 = 1
    else:
        norm_scale2 = np.sqrt(2) / max_distance2

    x1_normalized = x1_shifted * norm_scale1
    x2_normalized = x2_shifted * norm_scale2

    
    # TODO: Similarity transform 1 between x1_norm and x1
    T1 = computeH(x1_normalized, x1)

    # TODO: Similarity transform 2 between x2_norm and x2
    T2 = computeH(x2_normalized, x2)

    # TODO: Compute homography (which is x1 = T1^-1 H * T2 x2)
    H = computeH(x1_normalized, x2_normalized)

    # TODO: Denormalization
    H2to1 = np.matmul( np.matmul(np.linalg.inv(T1), H), T2)

    return H2to1




def computeH_ransac(locs1, locs2, opts):
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier
    bestH2to1 = None
    best_inliers = 0

    for _ in range(max_iters):
        # Randomly select 4 point correspondences
        random_indices = np.random.randint(0, len(locs1), 4)
        rand1 = locs1[random_indices]
        rand2 = locs2[random_indices]

        # Compute the homography H using the selected random points
        H = computeH(rand1, rand2)

        # Compute inliers based on the current homography H
        # x1 is source and x2 is destination
        x1 = np.concatenate((locs1, np.ones((len(locs1), 1))), axis=1)
        x2 = np.concatenate((locs2, np.ones((len(locs2), 1))), axis=1)

        # x2 = H * x1
        x2_pred = np.matmul(H, x1.T).T

        # Normalize the predicted points
        x2_pred = x2_pred / x2_pred[:, 2].reshape(-1, 1)

        # Compute the distance between the predicted points and the actual points
        distance = np.linalg.norm(x2 - x2_pred, axis=1)

        # Compute the number of inliers
        inliers = np.sum(distance < inlier_tol)

        # Update the best homography and the number of inliers
        if inliers > best_inliers:
            best_inliers = inliers
            bestH2to1 = H
        
    
    return bestH2to1, best_inliers



def compositeH(H2to1, template, img):
    
    #Create a composite image after warping the template image on top
    #of the image using the homography

    #Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    #For warping the template to the image, we need to invert it.

    # TODO: Create mask of same size as template
    mask = np.ones(template.shape)

    # TODO: Warp mask by appropriate homography
    warped_mask = cv2.warpPerspective(mask, H2to1, (img.shape[1], img.shape[0]))

    # TODO: Warp template by appropriate homography
    warped_template = cv2.warpPerspective(template, H2to1, (img.shape[1], img.shape[0]))

    # TODO: Use mask to combine the warped template and the image
    # Invert the mask so that the region of interest is 0 pixel value
    # Divide by 255 to convert the masks to the range [0, 1]
    inverted_mask = 1 - warped_mask
    composite_img = img * (inverted_mask/255) + warped_template * (warped_mask/255)
    
    return composite_img


