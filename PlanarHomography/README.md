## Planar Homography

### Overview
This repository contains Python code and datasets for implementing planar homography, a foundational concept in computer vision. The project explores the theoretical and practical aspects of planar homography, particularly in the context of augmented reality applications.

### Contents
- Python code for various stages of planar homography implementation.
- Datasets used for testing and demonstrating homography.

### Key Concepts
1. Planar Homographies as a Warp: Understanding the fundamental assumption of points lying on a plane and how pixel coordinates in one view can be mapped to another.

2. Direct Linear Transform: Techniques for solving homographies using linear algebra methods.

3. Matrix Decompositions: Exploring Eigenvalue and Singular Value Decomposition for calculating homographies.

4. Feature Detection and Matching: Techniques for identifying and matching feature points across different images.

5. Homography Computation and Normalization: Methods to compute and normalize homographies for stability and accuracy.

6. RANSAC for Homography Estimation: Implementing RANSAC to fit homography models to noisy data.

7. Application in Augmented Reality: Utilizing homographies to create augmented reality applications, including video incorporation and real-time AR.

8. Creating Panoramas: Techniques for stitching images together to create panoramas using homography.

### Applications
The codebase supports a range of applications from basic image warping and transformations to complex AR scenarios. Specific applications include augmented reality overlays, panorama creation, and feature-based image matching.

### Future Updates
Detailed descriptions and usage instructions for each script will be added soon.

#### Dependencies
Refer to `requirements.txt`. Tested with

* python==3.8.17
* numpy==1.21.2
* opencv-python==4.8.0.76
* scikit-image==0.21.0
* matplotlib==3.7.2