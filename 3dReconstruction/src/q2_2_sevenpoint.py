import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF

# Insert your package here


"""
Q2.2: Seven Point Algorithm for calculating the fundamental matrix
    Input:  pts1, 7x2 Matrix containing the corresponding points from image1
            pts2, 7x2 Matrix containing the corresponding points from image2
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated 3x3 fundamental matrixes.
    
    HINTS:
    (1) Normalize the input pts1 and pts2 scale paramter M.
    (2) Setup the seven point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2)
    (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        det(F) = 0. Solving this polynomial will give you one or three real solutions of the fundamental matrix. 
        Use np.polynomial.polynomial.polyroots to solve for the roots
    (6) Unscale the fundamental matrixes and return as Farray
"""


def sevenpoint(pts1, pts2, M):
    Farray = []

    # assert len(pts1) == len(pts2) == 7
    
    # Compute A matrix using the same method as in eightpoint
    AF = np.zeros((pts1.shape[0], 9))
    for i in range(pts1.shape[0]):
        # AF = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
        AF[i, :] = [ pts2[i,0]*pts1[i,0] , pts2[i,0]*pts1[i,1] , pts2[i,0], 
                    pts2[i,1]*pts1[i,0] , pts2[i,1]*pts1[i,1] , pts2[i,1], 
                    pts1[i,0], pts1[i,1], 1  ]
    
    # Solving for nullspace of A to get two Fs
    _, _, vt = np.linalg.svd(AF)
    fvec1 = vt[7]
    fvec2 = vt[8]
    
    Fmat = [np.array([  [fvec1[0], fvec1[3], fvec1[6]],  [fvec1[1], fvec1[4], fvec1[7]], [fvec1[2], fvec1[5], fvec1[8]] ]), 
            np.array([  [fvec2[0], fvec2[3], fvec2[6]],  [fvec2[1], fvec2[4], fvec2[7]], [fvec2[2], fvec2[5], fvec2[8]] ])]
    
    # Find F that meets the singularity constraint
    D = np.zeros((2, 2, 2))
    for i1 in range(2):        
        for i2 in range(2):
            for i3 in range(2):
                Dtmp = np.array([Fmat[i1][:, 0], Fmat[i2][:, 1], Fmat[i3][:, 2]]).T
                D[i1, i2, i3] = np.linalg.det(Dtmp)

    coefficients = np.array([
        -D[1, 0, 0] + D[0, 1, 1] + D[0, 0, 0] + D[1, 1, 0] + D[1, 0, 1] - D[0, 1, 0] - D[0, 0, 1] - D[1, 1, 1],
        D[0, 0, 1] - 2*D[0, 1, 1] - 2*D[1, 0, 1] + D[1, 0, 0] - 2*D[1, 1, 0] + D[0, 1, 0] + 3*D[1, 1, 1],
        D[1, 1, 0] + D[0, 1, 1] + D[1, 0, 1] - 3*D[1, 1, 1],
        D[1, 1, 1]
    ])

    roots = np.roots(coefficients)

    for r in roots:
        if np.isreal(r):
            Ftmp = r.real * Fmat[0] + (1 - r.real) * Fmat[1]
            Ftmp = np.array(Ftmp)
            F = refineF(Ftmp, pts1, pts2)
            F = F / F[2, 2]
            Farray.append(F)
    
    return Farray


if __name__ == "__main__":
    correspondence = np.load("data/some_corresp.npz")  # Loading correspondences
    intrinsics = np.load("data/intrinsics.npz")  # Loading the intrinscis of the camera
    K1, K2 = intrinsics["K1"], intrinsics["K2"]
    pts1, pts2 = correspondence["pts1"], correspondence["pts2"]
    im1 = plt.imread("data/im1.png")
    im2 = plt.imread("data/im2.png")

    # indices = np.arange(pts1.shape[0])
    # indices = np.random.choice(indices, 7, False)
    indices = np.array([82, 19, 56, 84, 54, 24, 18])

    M = np.max([*im1.shape, *im2.shape])

    Farray = sevenpoint(pts1[indices, :], pts2[indices, :], M)

    F = Farray[0]

    np.savez("q2_2.npz", F, M)

    # fundamental matrix must have rank 2!
    assert(np.linalg.matrix_rank(F) == 2)
    displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution.
#    np.random.seed(1)  # Added for testing, can be commented out

    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 500
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M = np.max([*im1.shape, *im2.shape])
    for i in range(max_iter):
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo, pts2_homo, F)
            F_res.append(F)
            ress.append(np.mean(res))

    min_idx = np.argmin(np.abs(np.array(ress)))
    F = F_res[min_idx]
    print(F)
    print("Error:", ress[min_idx])

    assert F.shape == (3, 3)
    assert F[2, 2] == 1
    assert np.linalg.matrix_rank(F) == 2
    assert np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1

