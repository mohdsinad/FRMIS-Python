"""
Core Stitching Module for FRMIS Algorithm

This module implements the main coordination logic for the Fast Robust Multi-Image Stitching
(FRMIS) algorithm. It orchestrates the complete stitching pipeline from input images to
final panoramic output.

The FRMIS pipeline consists of four main stages:
1. Grid Organization: Arrange input images into a logical grid structure
2. Pairwise Alignment: Compute initial alignments between adjacent image pairs
3. Global Alignment: Optimize global positions using Minimum Spanning Tree
4. Mosaic Assembly: Blend aligned images into final panoramic output

This implementation provides a complete end-to-end solution for large-scale image mosaicking
with robust error handling and performance monitoring.
"""

import os
import cv2
import time
import numpy as np

from natsort import natsorted
from src.pairwise_alignment import Pairwise_Alignment
from src.global_alignment import Global_Alignment


class FRMIS():
    """
    Fast Robust Multi-Image Stitching (FRMIS) main coordination class.
    
    This class implements the complete FRMIS pipeline, coordinating all stages of the
    image stitching process from input to final output. It manages the flow between
    pairwise alignment, global optimization, and final mosaic assembly.
    
    The class is designed to handle large grids of overlapping images efficiently,
    with built-in error handling and performance monitoring capabilities.
    
    Pipeline Overview:
    1. Grid Creation: Organize input images into a structured grid layout
    2. Pairwise Alignment: Use SIFT+RANSAC to align adjacent image pairs  
    3. Weight Computation: Calculate confidence weights based on alignment quality
    4. Global Alignment: Apply MST optimization for globally consistent positions
    5. Mosaic Assembly: Blend aligned images into final panoramic output
    """
    
    def __init__(self, args):
        """
        Initialize FRMIS stitcher with configuration parameters.
        
        Args:
            args: Configuration object containing all necessary parameters:
                - DATASET_DIR: Path to directory containing input images
                - OUTPUT_PATH: Path for final stitched mosaic output
                - MOSAIC_HEIGHT: Number of rows in image grid
                - MOSAIC_WIDTH: Number of columns in image grid
                - IMAGE_HEIGHT: Height of individual images in pixels
                - IMAGE_WIDTH: Width of individual images in pixels
                - IMAGE_CHANNELS: Number of color channels (1=grayscale, 3=RGB)
                - OVERLAP_X: Expected horizontal overlap between images in pixels
                - OVERLAP_Y: Expected vertical overlap between images in pixels
                - BLEND: Blending method for final assembly ('overlay', 'linear', etc.)
        """
        self.dataset_dir = args.DATASET_DIR
        self.output_path = args.OUTPUT_PATH
        self.grid_height = args.MOSAIC_HEIGHT
        self.grid_width = args.MOSAIC_WIDTH
        self.image_height = args.IMAGE_HEIGHT
        self.image_width = args.IMAGE_WIDTH
        self.image_channels = args.IMAGE_CHANNELS
        self.overlap_x = args.OVERLAP_X
        self.overlap_y = args.OVERLAP_Y
        self.blend_method = args.BLEND

    def _create_grid(self):
        """
        Organize input images into a logical grid structure.
        
        This method reads all image files from the dataset directory and arranges them
        into a 2D grid based on natural sorting order. The grid structure is essential
        for determining spatial relationships between adjacent images.
        
        Returns:
            numpy.ndarray: 2D array of image filenames arranged in grid layout
                          Shape: (grid_height, grid_width)
        
        Note:
            - Uses natural sorting to handle numbered filenames correctly (e.g., img1, img2, img10)
            - Assumes images are named in row-major order (left-to-right, top-to-bottom)
            - Total images must equal grid_height × grid_width
        """
        # Get all image files from dataset directory and sort them naturally
        # Natural sorting ensures proper ordering: img1.jpg, img2.jpg, ..., img10.jpg, img11.jpg
        sorted_images = natsorted(os.listdir(self.dataset_dir))
        
        # Reshape into 2D grid structure for spatial organization
        # This creates the logical mapping: grid[i][j] = image at row i, column j
        image_grid = np.array(sorted_images).reshape((self.grid_height, self.grid_width))
        return image_grid

    def _pairwise_alignment(self, image_grid):
        """
        Perform pairwise alignment between all adjacent image pairs.
        
        This method creates a Pairwise_Alignment object and computes translation
        transformations between all adjacent images in the grid using SIFT features
        and RANSAC-based robust estimation.
        
        Args:
            image_grid (numpy.ndarray): 2D array of image filenames in grid layout
            
        Returns:
            dict: Pairwise alignment results containing:
                - tx_north, ty_north: Translation vectors for vertical connections
                - tx_west, ty_west: Translation vectors for horizontal connections
                - inliers_north, inliers_west: Quality metrics (number of inliers)
        
        Note:
            - Only processes adjacent pairs (north and west connections)
            - Failed alignments are marked with NaN values
            - Results serve as input for global alignment optimization
        """
        # Create and run pairwise alignment on overlap regions using SIFT + RANSAC
        return Pairwise_Alignment(
            dataset_dir=self.dataset_dir,
            image_grid=image_grid,
            image_height = self.image_height,
            image_width = self.image_width,
            overlap_x = self.overlap_x,
            overlap_y = self.overlap_y
        ).align()
    
    def _compute_weights(self, translations):
        """
        Compute confidence weights for global alignment based on pairwise alignment quality.
        
        This method converts the number of inlier matches from pairwise alignment into
        confidence weights for the global optimization. Higher numbers of inliers
        indicate more reliable alignments and receive lower weights (higher confidence).
        
        Args:
            translations (dict): Pairwise alignment results containing inlier counts
            
        Returns:
            tuple: (weight_north, weight_west) - Normalized confidence weights
                  Lower weights indicate higher confidence in the alignment
        
        Algorithm:
            1. Compute inverse of inlier counts (more inliers = lower weight = higher confidence)
            2. Normalize weights to [0,1] range for numerical stability
            3. NaN values indicate failed alignments (infinite weight)
        """
        # Convert inlier counts to weights: more inliers = lower weight = higher confidence
        # This creates a cost function where reliable alignments have lower cost
        weight_north = 1.0 / translations['inliers_north']
        weight_west = 1.0 / translations['inliers_west']

        # Normalize weights to [0,1] range to prevent numerical issues in MST algorithm
        # This ensures all weights are comparable regardless of absolute inlier counts
        max_weight = max(np.nanmax(weight_north), np.nanmax(weight_west))
        weight_north = weight_north / max_weight
        weight_west = weight_west / max_weight

        # Handle NaN values
        weight_north = np.nan_to_num(weight_north, nan=1)
        weight_west = np.nan_to_num(weight_west, nan=1)

        return weight_north, weight_west
    
    def _global_alignment(self, translations, weight_north, weight_west):
        """
        Perform global alignment optimization using Minimum Spanning Tree.
        
        This method creates a Global_Alignment object and runs the MST algorithm
        to compute globally optimal positions for all image tiles. The MST approach
        minimizes cumulative alignment errors across the entire mosaic.
        
        Args:
            translations (dict): Pairwise translation vectors between adjacent images
            weight_north (numpy.ndarray): Confidence weights for vertical connections
            weight_west (numpy.ndarray): Confidence weights for horizontal connections
            
        Returns:
            tuple: (global_y_pos, global_x_pos) - Global positions for each image tile
        
        Note:
            - Uses Prim's algorithm to build MST from most confident alignments
            - Unconnected tiles receive NaN positions
            - Results provide pixel coordinates for final mosaic assembly
        """
        return Global_Alignment(
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            translations=translations,
            weight_north=weight_north,
            weight_west=weight_west
        ).mst()
    
    def _compute_blend_weights(self):
        """
        Computes a linear pixel weight matrix based on distance to the image center.
        The weight is based on the minimum distance to any edge.
        """
    
        # 1. Compute distance to the nearest horizontal and vertical edge
        # d_min_mat_i: [height x 1] vector
        d_min_mat_i = np.minimum(np.arange(1, self.image_height + 1), self.image_height - np.arange(self.image_height) + 1).reshape(self.image_height, 1)
        # d_min_mat_j: [1 x width] vector
        d_min_mat_j = np.minimum(np.arange(1, self.image_width + 1), self.image_width - np.arange(self.image_width) + 1).reshape(1, self.image_width)

        # 2. Compute the minimum distance matrix
        # w_matt: [height x width] matrix (Outer product)
        w_matt = d_min_mat_i * d_min_mat_j
    
        # 3. Apply alpha power (for non-linear weighting)
        w_matt = w_matt.astype(np.float32) ** 1.5
        
        # Replicate the 2D weights across the 3 color channels
        w_mat = np.stack([w_matt] * 3, axis=-1)
 
        return w_mat

    def assemble_grid(self, image_grid, global_y_pos, global_x_pos):
        """
        Assemble the final mosaic by placing images at their computed global positions.
        
        This method creates the final panoramic image by loading each image and placing
        it at its globally optimized position. Currently implements simple overlay blending
        where later images overwrite earlier ones in overlapping regions.
        
        Args:
            image_grid (numpy.ndarray): 2D array of image filenames
            global_y_pos (numpy.ndarray): Y coordinates for each image tile
            global_x_pos (numpy.ndarray): X coordinates for each image tile
            
        Note:
            - Images with NaN positions are skipped (unconnected tiles)
            - Uses simple overlay blending (last image wins in overlaps)
            - Automatically handles boundary clipping for mosaic edges
            - Saves final result to self.output_path
        """
        # Calculate required mosaic dimensions based on image positions and sizes
        # Add image dimensions to positions to account for full image extents

        mosaic_height = int(np.nanmax(global_y_pos + self.image_height))
        mosaic_width = int(np.nanmax(global_x_pos + self.image_width))

        # Initialize empty mosaic canvas with specified number of channels
        # RGB images: 3 channels, grayscale: 1 channel
        mosaic = np.zeros((mosaic_height, mosaic_width, self.image_channels), dtype=np.float32)

        if self.blend_method.lower() == 'overlay':
            # Process each image in the grid and place it at computed position
            for i in range(self.grid_height):
                for j in range(self.grid_width):
                    # Skip images that couldn't be positioned (NaN coordinates)
                    if not np.isnan(global_y_pos[i, j]):
                        # Load image in color mode for final assembly
                        img = cv2.imread(os.path.join(self.dataset_dir, image_grid[i][j]), cv2.IMREAD_COLOR)

                        # Calculate placement coordinates in mosaic
                        y_start = int(global_y_pos[i, j])
                        x_start = int(global_x_pos[i, j])
                        y_end = y_start + self.image_height
                        x_end = x_start + self.image_width

                        # Skip images that fall completely outside mosaic bounds
                        if y_start >= mosaic_height or x_start >= mosaic_width:
                            continue
                        
                        # Clip coordinates to mosaic boundaries to handle edge cases
                        y0 = max(0, y_start)              # Top boundary
                        x0 = max(0, x_start)              # Left boundary  
                        y1 = min(y_end, mosaic_height)    # Bottom boundary
                        x1 = min(x_end, mosaic_width)     # Right boundary
                        h = y1 - y0                       # Clipped height
                        w = x1 - x0                       # Clipped width
                        
                        # Skip degenerate cases where clipping results in zero area
                        if h <= 0 or w <= 0:
                            continue
                                    
                        # Calculate corresponding crop region in source image
                        # Handle cases where image extends beyond mosaic boundaries
                        img_y0 = 0 if y_start >= 0 else -y_start  # Top crop offset
                        img_x0 = 0 if x_start >= 0 else -x_start  # Left crop offset
                        
                        # Extract the relevant portion of the source image
                        tile = img[img_y0:img_y0+h, img_x0:img_x0+w]
                        
                        # Place tile in mosaic using simple overlay blending
                        mosaic[y0:y1, x0:x1] = tile
        
        elif self.blend_method.lower() == 'linear':
            # Initialize weight matrix for linear blending
            w_mat = self._compute_blend_weights()

            # Initialize canvas for accumulating weights (The Denominator)
            weight_sum = np.zeros_like(mosaic, dtype=np.float32) 
            
            # Process each image in the grid and place it at computed position
            for i in range(self.grid_height):
                for j in range(self.grid_width):
                    # Skip images that couldn't be positioned (NaN tiling indicator)
                    if not np.isnan(global_y_pos[i, j]):
                        # Load image in color mode for final assembly
                        img = cv2.imread(os.path.join(self.dataset_dir, image_grid[i][j]), cv2.IMREAD_COLOR)

                        # Calculate placement coordinates in mosaic
                        y_start = int(global_y_pos[i, j])
                        x_start = int(global_x_pos[i, j])
                        y_end = y_start + self.image_height
                        x_end = x_start + self.image_width

                        # Skip images that fall completely outside mosaic bounds
                        if y_start >= mosaic_height or x_start >= mosaic_width:
                            continue
                        
                        # Clip coordinates to mosaic boundaries to handle edge cases
                        y0 = max(0, y_start)              # Top boundary
                        x0 = max(0, x_start)              # Left boundary  
                        y1 = min(y_end, mosaic_height)    # Bottom boundary
                        x1 = min(x_end, mosaic_width)     # Right boundary
                        h = y1 - y0                       # Clipped height
                        w = x1 - x0                       # Clipped width
                        
                        # Skip degenerate cases where clipping results in zero area
                        if h <= 0 or w <= 0:
                            continue
                                    
                        # Calculate corresponding crop region in source image
                        # Handle cases where image extends beyond mosaic boundaries
                        img_y0 = 0 if y_start >= 0 else -y_start  # Top crop offset
                        img_x0 = 0 if x_start >= 0 else -x_start  # Left crop offset

                        # Apply weights
                        weighted_image = img.astype(np.float32) * w_mat

                        # Extract the relevant weight portion
                        weight_tile = w_mat[img_y0:img_y0+h, img_x0:img_x0+w]                         

                        # Extract the relevant, weighted tile portion
                        weighted_tile = weighted_image[img_y0:img_y0+h, img_x0:img_x0+w]
                        
                        # Accumulation (Summation)
                        # I(y_st:y_end,x_st:x_end,:) = I(y_st:y_end,x_st:x_end,:) + current_image (weighted)
                        mosaic[y0:y1, x0:x1] += weighted_tile

                        # Accumulation of Weights (Denominator)
                        weight_sum[y0:y1, x0:x1] += weight_tile
            
            # Prevent division by zero where no images were placed
            weight_sum[weight_sum == 0] = 1.0
        
            # Calculate weighted average: (Sum of Weighted Pixels) / (Sum of Weights)
            mosaic = mosaic / weight_sum
            
            # Cast back to the original type (e.g., uint8)
            mosaic = mosaic.clip(0, 255).astype(np.uint8)
            
        # Save final mosaic to output path
        cv2.imwrite(self.output_path, mosaic.astype(np.uint8))

    def run(self):
        """
        Execute the complete FRMIS stitching pipeline.
        
        This method orchestrates the entire image stitching process from input images
        to final panoramic output. It coordinates all pipeline stages and provides
        performance timing information for each major component.
        
        Returns:
            dict: Performance timing results containing:
                - time_pairwise: Time spent on pairwise alignment (seconds)
                - time_global: Time spent on global alignment optimization (seconds)
                - time_assembly: Time spent on final mosaic assembly (seconds)
        
        Pipeline Stages:
            1. Grid Creation: Organize input images into spatial grid
            2. Pairwise Alignment: Compute translations between adjacent pairs
            3. Weight Computation: Calculate confidence weights from alignment quality
            4. Global Alignment: Optimize positions using MST algorithm
            5. Mosaic Assembly: Blend images into final panoramic output
        
        Note:
            - Each stage is timed independently for performance analysis
            - Failed alignments are handled gracefully with NaN values
            - Final mosaic is automatically saved to configured output path
        """
        # Stage 1: Create logical grid structure from input images
        print("Stage 1: Creating image grid...")
        image_grid = self._create_grid()
        print(f"Organized {self.grid_height}×{self.grid_width} = {self.grid_height*self.grid_width} images")

        # Stage 2: Perform pairwise alignment between adjacent images
        print("Stage 2: Computing pairwise alignments...")
        time_pairwise = time.time()
        translations = self._pairwise_alignment(image_grid)
        time_pairwise = time.time() - time_pairwise
        print(f"Pairwise alignment completed in {time_pairwise:.2f} seconds")

        # Stage 3: Compute confidence weights from alignment quality
        print("Stage 3: Computing confidence weights...")
        weight_north, weight_west = self._compute_weights(translations)

        # Stage 4: Perform global alignment optimization using MST
        print("Stage 4: Optimizing global positions...")
        time_global = time.time()
        global_y_pos, global_x_pos = self._global_alignment(translations, weight_north, weight_west)
        time_global = time.time() - time_global
        print(f"Global alignment completed in {time_global:.2f} seconds")

        # Stage 5: Assemble final mosaic from aligned images
        print("Stage 5: Assembling final mosaic...")
        time_assembly = time.time()
        self.assemble_grid(image_grid, global_y_pos, global_x_pos)
        time_assembly = time.time() - time_assembly
        print(f"Mosaic assembly completed in {time_assembly:.2f} seconds")
        print(f"Final mosaic saved to: {self.output_path}")

