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
- Multiprocessing support for parallel execution
"""

import os
import cv2
import numpy as np
import multiprocessing

def detect_and_match(roi_image_1, roi_image_2, x1, y1, x2, y2):
    """
    Detect SIFT features and compute robust translation between two image regions.
    Standalone function to be picklable for multiprocessing.
    """
    # Initialize SIFT detector for feature detection
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both image regions
    kp1, desc1 = sift.detectAndCompute(roi_image_1, None)
    kp2, desc2 = sift.detectAndCompute(roi_image_2, None)

    # Validate that descriptors were successfully computed
    if desc1 is None or desc2 is None:
        return np.array([]), np.array([]), 0, False
    if len(desc1) == 0 or len(desc2) == 0:
        return np.array([]), np.array([]), 0, False
    # FLANN with k=2 (kNN matching) requires at least 2 descriptors in training set
    if desc1.shape[0] < 2 or desc2.shape[0] < 2:
        return np.array([]), np.array([]), 0, False

    # Match features using FLANN
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

    if not good_matches:
        return np.array([]), np.array([]), 0, False

    # Extract matched point coordinates and adjust for ROI offsets
    matched_points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]) + np.array([x1, y1])
    matched_points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]) + np.array([x2, y2])

    # Use RANSAC to robustly estimate affine transformation
    translation, inliers = cv2.estimateAffinePartial2D(
        matched_points1, matched_points2,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=99.99 / 100.0
        )
    
    if translation is None:
        return np.array([]), np.array([]), 0, False
    
    tx = translation[0, 2]
    ty = translation[1, 2]

    if inliers.sum() >= 1:
        success = True
    else:
        success = False

    return tx, ty, inliers.sum(), success

def run_alignment_task(task):
    """
    Worker function to process a single alignment task.
    """
    task_type = task['type']
    image_1_path = task['image_1_path']
    image_2_path = task['image_2_path']
    params = task['params']
    
    image_height = params['image_height']
    image_width = params['image_width']
    overlap_x = params['overlap_x']
    overlap_y = params['overlap_y']

    # Load images
    image_1 = cv2.imread(image_1_path, cv2.IMREAD_GRAYSCALE)
    image_2 = cv2.imread(image_2_path, cv2.IMREAD_GRAYSCALE)

    if image_1 is None or image_2 is None:
        return None

    image_1 = cv2.resize(image_1, (1280, 720), interpolation=cv2.INTER_AREA)
    image_2 = cv2.resize(image_2, (1280, 720), interpolation=cv2.INTER_AREA)

    if task_type == 'north':
        # ROI setup for north alignment
        x1, y1, w1, h1 = (0, 0, image_width, overlap_y)
        roi_image_1 = image_1[y1:y1+h1, x1:x1+w1]
        
        x2, y2, w2, h2 = (0, (image_height-overlap_y), image_width, overlap_y)
        roi_image_2 = image_2[y2:y2+h2, x2:x2+w2]
        
        return detect_and_match(roi_image_1, roi_image_2, x1, y1, x2, y2)

    elif task_type == 'west':
        # ROI setup for west alignment
        x1, y1, w1, h1 = (0, 0, overlap_x, image_height) 
        roi_image_1 = image_1[y1:y1+h1, x1:x1+w1]

        x2, y2, w2, h2 = ((image_width-overlap_x), 0, overlap_x, image_height)
        roi_image_2 = image_2[y2:y2+h2, x2:x2+w2]

        return detect_and_match(roi_image_1, roi_image_2, x1, y1, x2, y2)
    
    return None

class Pairwise_Alignment():
    """
    Pairwise alignment class for computing translation transformations between adjacent images.
    """
    
    def __init__(self, dataset_dir, image_grid, image_height, image_width, overlap_x, overlap_y):
        self.dataset_dir = dataset_dir
        self.image_grid = image_grid
        self.grid_height = image_grid.shape[0]
        self.grid_width = image_grid.shape[1]
        self.image_height = image_height
        self.image_width = image_width
        self.overlap_x = overlap_x
        self.overlap_y = overlap_y

        self.tx_north = np.full((self.grid_height, self.grid_width), np.nan)
        self.ty_north = np.full((self.grid_height, self.grid_width), np.nan)
        self.inliers_north = np.full((self.grid_height, self.grid_width), np.nan)

        self.tx_west = np.full((self.grid_height, self.grid_width), np.nan)
        self.ty_west = np.full((self.grid_height, self.grid_width), np.nan)
        self.inliers_west = np.full((self.grid_height, self.grid_width), np.nan)

    def align(self):
        """
        Perform pairwise alignment for all adjacent image pairs in the grid using multiprocessing.
        """
        tasks = []
        params = {
            'image_height': self.image_height,
            'image_width': self.image_width,
            'overlap_x': self.overlap_x,
            'overlap_y': self.overlap_y
        }

        # Prepare tasks
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                # North alignment task
                if i > 0:
                    tasks.append({
                        'type': 'north',
                        'i': i, 'j': j,
                        'image_1_path': os.path.join(self.dataset_dir, self.image_grid[i][j]),
                        'image_2_path': os.path.join(self.dataset_dir, self.image_grid[i-1][j]),
                        'params': params
                    })

                # West alignment task
                if j > 0:
                    tasks.append({
                        'type': 'west',
                        'i': i, 'j': j,
                        'image_1_path': os.path.join(self.dataset_dir, self.image_grid[i][j]),
                        'image_2_path': os.path.join(self.dataset_dir, self.image_grid[i][j-1]),
                        'params': params
                    })

        # Execute tasks in parallel
        # Use number of CPU cores for pool size
        num_processes = multiprocessing.cpu_count()
        print(f"Starting pairwise alignment with {num_processes} processes...")
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(run_alignment_task, tasks)

        # Process results
        for task, result in zip(tasks, results):
            if result is None:
                continue
                
            tx, ty, num_inliers, success = result
            i, j = task['i'], task['j']
            
            print(f"Processed {task['type']} alignment for ({i}, {j}): success={success}")

            if success:
                if task['type'] == 'north':
                    self.tx_north[i, j] = tx
                    self.ty_north[i, j] = ty
                    self.inliers_north[i, j] = num_inliers
                elif task['type'] == 'west':
                    self.tx_west[i, j] = tx
                    self.ty_west[i, j] = ty
                    self.inliers_west[i, j] = num_inliers

        return {
            'tx_north': self.tx_north, 'ty_north': self.ty_north,
            'tx_west': self.tx_west, 'ty_west': self.ty_west,
            'inliers_north': self.inliers_north,
            'inliers_west': self.inliers_west
        }

