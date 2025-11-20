"""
Pairwise Alignment Module for Image Stitching

This module implements the pairwise alignment step of the FRMIS algorithm, which computes
initial alignment transformations between adjacent image pairs in a grid arrangement.
The module uses SIFT feature detection and RANSAC-based robust estimation to find
translation parameters between overlapping images.

Key Features:
- SIFT feature detection for robust keypoint matching
- FLANN-based efficient feature matching
- RANSAC for outlier rejection and robust transformation estimation
- Separate handling of vertical (north) and horizontal (west) connections
- Region of Interest (ROI) processing for improved alignment accuracy
"""

import os
import cv2
import numpy as np


class Pairwise_Alignment():
    """
    Pairwise alignment class for computing translation transformations between adjacent images.
    
    This class processes images in a grid arrangement and computes pairwise translation
    transformations between adjacent images using SIFT features and RANSAC estimation.
    The alignment is performed separately for vertical (north) and horizontal (west) 
    connections to optimize the overlap regions.
    
    Attributes:
        dataset_dir (str): Directory containing the input images
        image_grid (numpy.ndarray): 2D array containing image filenames in grid arrangement
        grid_height (int): Number of rows in the image grid
        grid_width (int): Number of columns in the image grid  
        image_height (int): Height of individual images in pixels
        image_width (int): Width of individual images in pixels
        overlap_x (int): Horizontal overlap between adjacent images in pixels
        overlap_y (int): Vertical overlap between adjacent images in pixels
    """
    
    def __init__(self, dataset_dir, image_grid, image_height, image_width, overlap_x, overlap_y):
        """
        Initialize the Pairwise_Alignment object with grid configuration and image parameters.
        
        Args:
            dataset_dir (str): Path to directory containing input images
            image_grid (numpy.ndarray): 2D array of image filenames arranged in grid layout
            image_height (int): Height of individual images in pixels
            image_width (int): Width of individual images in pixels  
            overlap_x (int): Expected horizontal overlap between adjacent images in pixels
            overlap_y (int): Expected vertical overlap between adjacent images in pixels
        
        Note:
            - All images are assumed to have the same dimensions
            - Overlap values should be positive and less than image dimensions
            - Grid coordinates: (0,0) is top-left, (i,j) represents row i, column j
        """
        self.dataset_dir = dataset_dir
        self.image_grid = image_grid
        self.grid_height = image_grid.shape[0]
        self.grid_width = image_grid.shape[1]
        self.image_height = image_height
        self.image_width = image_width
        self.overlap_x = overlap_x
        self.overlap_y = overlap_y

        # Initialize translation arrays for north (vertical) connections
        # tx_north[i,j] = X translation from image (i-1,j) to image (i,j)
        # ty_north[i,j] = Y translation from image (i-1,j) to image (i,j)
        self.tx_north = np.full((self.grid_height, self.grid_width), np.nan)
        self.ty_north = np.full((self.grid_height, self.grid_width), np.nan)
        self.inliers_north = np.full((self.grid_height, self.grid_width), np.nan)

        # Initialize translation arrays for west (horizontal) connections  
        # tx_west[i,j] = X translation from image (i,j-1) to image (i,j)
        # ty_west[i,j] = Y translation from image (i,j-1) to image (i,j)
        self.tx_west = np.full((self.grid_height, self.grid_width), np.nan)
        self.ty_west = np.full((self.grid_height, self.grid_width), np.nan)
        self.inliers_west = np.full((self.grid_height, self.grid_width), np.nan)

    def _detect_and_match(self, roi_image_1, roi_image_2, x1, y1, x2, y2):
        """
        Detect SIFT features and compute robust translation between two image regions.
        
        This method performs the core feature detection and matching pipeline:
        1. Detects SIFT features in both image regions
        2. Matches features using FLANN-based matcher
        3. Applies Lowe's ratio test for robust matching
        4. Uses RANSAC to estimate translation and reject outliers
        
        Args:
            roi_image_1 (numpy.ndarray): First image region (grayscale)
            roi_image_2 (numpy.ndarray): Second image region (grayscale)
            x1, y1 (int): Top-left coordinates of roi_image_1 in full image
            x2, y2 (int): Top-left coordinates of roi_image_2 in full image
            
        Returns:
            tuple: (tx, ty, num_inliers, success) where:
                - tx (float): X translation in pixels
                - ty (float): Y translation in pixels  
                - num_inliers (int): Number of inlier matches after RANSAC
                - success (bool): Whether alignment was successful
        
        Note:
            - Returns ([], [], 0, False) if alignment fails
            - Coordinates are adjusted for ROI offsets in full images
            - Uses affine transformation but extracts only translation component
        """
        # Initialize SIFT detector for feature detection
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors for both image regions
        kp1, desc1 = sift.detectAndCompute(roi_image_1, None)
        kp2, desc2 = sift.detectAndCompute(roi_image_2, None)

        # Validate that descriptors were successfully computed
        # This can fail if images have insufficient texture or contrast
        if desc1 is None or desc2 is None:
            return np.array([]), np.array([]), 0, False
        if len(desc1) == 0 or len(desc2) == 0:
            return np.array([]), np.array([]), 0, False
        # FLANN with k=2 (kNN matching) requires at least 2 descriptors in training set
        if desc1.shape[0] < 2 or desc2.shape[0] < 2:
            return np.array([]), np.array([]), 0, False

        # Match features using FLANN (Fast Library for Approximate Nearest Neighbors)
        # This is more efficient than brute-force matching for large descriptor sets
        index_params = dict(algorithm=1, trees=5)  # KDTREE algorithm with 5 trees
        search_params = dict(checks=50)            # Maximum leaf checks during search
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)  # Find 2 nearest neighbors for each descriptor

        # Apply Lowe's ratio test to filter good matches
        # Good match: distance to closest neighbor < 0.5 * distance to second closest
        # This effectively removes ambiguous matches where multiple similar features exist
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:  # Ensure we have both nearest neighbors
                m, n = match_pair     # m = closest, n = second closest
                if m.distance < 0.7 * n.distance:  # Lowe's ratio test
                    good_matches.append(m)

        # Check if we have sufficient good matches for reliable estimation
        if good_matches == []:
            return np.array([]), np.array([]), 0, False

        # Extract matched point coordinates and adjust for ROI offsets in full images
        # This converts local ROI coordinates back to full image coordinates
        matched_points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]) + np.array([x1, y1])
        matched_points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]) + np.array([x2, y2])

        # Use RANSAC to robustly estimate affine transformation
        # RANSAC iteratively fits models to random subsets and finds the one with most inliers
        translation, inliers = cv2.estimateAffinePartial2D(
            matched_points1, matched_points2,    # Source and destination points
            method=cv2.RANSAC,                   # Use RANSAC for robust estimation
            ransacReprojThreshold=3.0,           # Maximum reprojection error for inliers (pixels)
            maxIters=2000,                       # Maximum RANSAC iterations
            confidence=99.99 / 100.0             # Desired confidence level (99.99%)
            )
        
        # Check if transformation estimation was successful
        if translation is None:
            return np.array([]), np.array([]), 0, False
        
        # Extract translation components from the 2x3 affine transformation matrix
        # translation = [[ cos(θ)*s  -sin(θ)*s   tx ]
        #               [ sin(θ)*s   cos(θ)*s   ty ]]
        # For pure translation: θ=0, s=1, so we extract tx and ty
        tx = translation[0, 2]  # X translation
        ty = translation[1, 2]  # Y translation

        # Consider alignment successful if we have at least 1 inlier
        # In practice, more inliers indicate higher confidence
        if inliers.sum() >= 1:
            success = True
        else:
            success = False

        return tx, ty, inliers.sum(), success

    def _compute_north_transform(self, image_1, image_2):
        """
        Compute vertical (north) alignment between two vertically adjacent images.
        
        This method aligns a current image with the image above it in the grid by
        comparing overlapping regions. It extracts the bottom overlap region from
        the upper image and the top overlap region from the lower image.
        
        Args:
            image_1 (numpy.ndarray): Current image (lower position in grid)
            image_2 (numpy.ndarray): Previous image (upper position in grid)
            
        Returns:
            tuple: (tx, ty, num_inliers, success) - Translation and quality metrics
            
        Note:
            - ROI from image_1: top portion (height = overlap_y)
            - ROI from image_2: bottom portion (height = overlap_y)
            - Positive ty typically indicates image_1 is shifted down relative to image_2
        """
        # Define ROI in current image (image_1): extract top overlap region
        # This region should match with the bottom of the image above
        x1, y1, w1, h1 = (0, 0, self.image_width, self.overlap_y)
        roi_image_1 = image_1[y1:y1+h1, x1:x1+w1]
        
        # Define ROI in previous image (image_2): extract bottom overlap region  
        # This region should match with the top of the image below
        x2, y2, w2, h2 = (0, (self.image_height-self.overlap_y), self.image_width, self.overlap_y)
        roi_image_2 = image_2[y2:y2+h2, x2:x2+w2]

        return self._detect_and_match(roi_image_1, roi_image_2, x1, y1, x2, y2)

    def _compute_west_transform(self, image_1, image_2):
        """
        Compute horizontal (west) alignment between two horizontally adjacent images.
        
        This method aligns a current image with the image to its left in the grid by
        comparing overlapping regions. It extracts the right overlap region from the
        left image and the left overlap region from the right image.
        
        Args:
            image_1 (numpy.ndarray): Current image (right position in grid)
            image_2 (numpy.ndarray): Previous image (left position in grid)
            
        Returns:
            tuple: (tx, ty, num_inliers, success) - Translation and quality metrics
            
        Note:
            - ROI from image_1: left portion (width = overlap_x)
            - ROI from image_2: right portion (width = overlap_x)  
            - Positive tx typically indicates image_1 is shifted right relative to image_2
        """
        # Define ROI in current image (image_1): extract left overlap region
        # This region should match with the right side of the image to the left
        x1, y1, w1, h1 = (0, 0, self.overlap_x, self.image_height) 
        roi_image_1 = image_1[y1:y1+h1, x1:x1+w1]

        # Define ROI in previous image (image_2): extract right overlap region
        # This region should match with the left side of the image to the right
        x2, y2, w2, h2 = ((self.image_width-self.overlap_x), 0, self.overlap_x, self.image_height)
        roi_image_2 = image_2[y2:y2+h2, x2:x2+w2]

        return self._detect_and_match(roi_image_1, roi_image_2, x1, y1, x2, y2)

    def align(self):
        """
        Perform pairwise alignment for all adjacent image pairs in the grid.
        
        This method processes each image in the grid and computes alignment transformations
        with its neighbors. For each image at position (i,j), it computes:
        - North alignment: with image at (i-1,j) if it exists
        - West alignment: with image at (i,j-1) if it exists
        
        The alignment results are stored in class attributes for later use by global alignment.
        
        Returns:
            dict: Dictionary containing all computed alignment results:
                - 'tx_north', 'ty_north': X,Y translations for vertical connections
                - 'tx_west', 'ty_west': X,Y translations for horizontal connections  
                - 'inliers_north', 'inliers_west': Number of inlier matches for quality assessment
                
        Note:
            - Failed alignments remain as NaN values in the result arrays
            - Progress information is printed during processing
            - Images are loaded as grayscale for feature detection
        """
        # Process each image in the grid row by row, column by column
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                print(f"Processing image at grid position ({i}, {j}): {self.image_grid[i][j]}")

                # Load the current image as grayscale for feature detection
                # Grayscale reduces computational complexity while preserving structural features
                image_1 = cv2.imread(os.path.join(self.dataset_dir, self.image_grid[i][j]), cv2.IMREAD_GRAYSCALE)
                
                # Compute north (vertical) alignment if not in the first row
                if i > 0:
                    # Load the image immediately above the current image
                    image_2 = cv2.imread(os.path.join(self.dataset_dir, self.image_grid[i-1][j]), cv2.IMREAD_GRAYSCALE)

                    # Compute translation from image above to current image
                    tx, ty, num_inliers, success = self._compute_north_transform(image_1, image_2)

                    print(f"  North alignment: tx={tx}, ty={ty}, inliers={num_inliers}, success={success}")

                    # Store results if alignment was successful
                    if success:
                        self.tx_north[i, j] = tx
                        self.ty_north[i, j] = ty
                        self.inliers_north[i, j] = num_inliers

                # Compute west (horizontal) alignment if not in the first column
                if j > 0:
                    # Load the image immediately to the left of the current image
                    image_2 = cv2.imread(os.path.join(self.dataset_dir, self.image_grid[i][j-1]), cv2.IMREAD_GRAYSCALE)

                    # Compute translation from image to the left to current image
                    tx, ty, num_inliers, success = self._compute_west_transform(image_1, image_2)

                    print(f"  West alignment: tx={tx}, ty={ty}, inliers={num_inliers}, success={success}")

                    # Store results if alignment was successful
                    if success:
                        self.tx_west[i, j] = tx
                        self.ty_west[i, j] = ty
                        self.inliers_west[i, j] = num_inliers

        # Return all computed alignment data as a dictionary
        # This data will be used by the global alignment stage
        return {
            'tx_north': self.tx_north, 'ty_north': self.ty_north,     # Vertical translations
            'tx_west': self.tx_west, 'ty_west': self.ty_west,         # Horizontal translations
            'inliers_north': self.inliers_north,                     # Vertical alignment quality
            'inliers_west': self.inliers_west                        # Horizontal alignment quality
        }
