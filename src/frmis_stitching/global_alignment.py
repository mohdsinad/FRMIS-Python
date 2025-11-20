"""
Global Alignment Module for Image Stitching using Minimum Spanning Tree (MST)

This module implements the global alignment step of the FRMIS (Fast Robust Multi-Image Stitching)
algorithm. It uses a Minimum Spanning Tree approach to determine the optimal global positions
of image tiles in a grid arrangement.

The implementation follows the MATLAB version of the algorithm, using Prim's algorithm to
build the MST and compute global tile positions based on pairwise translation estimates.
"""

import numpy as np


class Global_Alignment():
    """
    Global alignment class for computing optimal tile positions using MST algorithm.
    
    This class implements the minimum spanning tree approach for global alignment in image
    stitching. It takes pairwise alignment results (translations and weights) and computes
    the global position of each image tile to minimize alignment errors.
    """
    
    def __init__(self, grid_height, grid_width, translations, weight_north, weight_west):
        """
        Initialize the Global_Alignment object with grid configuration and alignment data.
        
        Args:
            grid_height (int): Number of rows in the image grid
            grid_width (int): Number of columns in the image grid
            translations (dict): Dictionary containing translation vectors:
                - 'tx_north': X translations for north (vertical) connections
                - 'ty_north': Y translations for north (vertical) connections  
                - 'tx_west': X translations for west (horizontal) connections
                - 'ty_west': Y translations for west (horizontal) connections
            weight_north (numpy.ndarray): Confidence weights for north connections (grid_height x grid_width)
            weight_west (numpy.ndarray): Confidence weights for west connections (grid_height x grid_width)
        
        Note:
            - Lower weights indicate better alignment confidence
            - NaN values in weights indicate no valid connection
            - Coordinate system: north = down, west = right in image coordinates
        """
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.translations = translations
        self.weight_north = weight_north
        self.weight_west = weight_west

    def mst(self):
        """
        Implementation of minimum spanning tree algorithm.Uses Prim's algorithm to
        iteratively build the MST by selecting minimum weight edges.
        
        This method implements the core MST algorithm that:
        1. Finds the starting tile with minimum weight among all connections
        2. Iteratively adds the minimum weight edge connecting tree vertices to non-tree vertices
        3. Computes global positions based on translation vectors as tiles are added
        4. Returns the final global positions for all connected tiles
        
        Returns:
            tuple: (global_y_pos, global_x_pos) - 2D arrays containing the global positions
                   of each tile in the stitched image coordinate system
        
        Algorithm Details:
            - Uses Prim's algorithm for MST construction
            - Coordinate system: (0,0) at top-left, Y increases downward, X increases rightward
            - North connections: from tile (i,j) to tile (i+1,j) - moving down
            - West connections: from tile (i,j) to tile (i,j+1) - moving right
            - Unconnected tiles have NaN positions
        """
        # Initialize global position arrays with zeros
        # These will store the computed positions of each tile in the final stitched image
        global_y_pos = np.zeros((self.grid_height, self.grid_width))
        global_x_pos = np.zeros((self.grid_height, self.grid_width))
        
        # Initialize boolean matrix to track which tiles are already connected to the MST
        # True = tile is in the MST, False = tile is not yet connected
        tiling_indicator = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        
        # Find starting tile: select the tile with the minimum weight among ALL connections
        # This ensures we start with the most confident alignment
        min_val_north = np.nanmin(self.weight_north)  # Min weight in north connections
        min_val_west = np.nanmin(self.weight_west)    # Min weight in west connections
        
        # Choose the global minimum between north and west connections
        if min_val_north < min_val_west:
            # Minimum is in north connections - find its position
            start_i, start_j = np.where(self.weight_north == min_val_north)
            start_i, start_j = start_i[0], start_j[0]
        else:
            # Minimum is in west connections - find its position
            start_i, start_j = np.where(self.weight_west == min_val_west)
            start_i, start_j = start_i[0], start_j[0]
        
        # Mark the starting tile as connected to the MST
        # This tile will have position (0,0) initially
        tiling_indicator[start_i, start_j] = True
        
        # Iteratively build the MST using Prim's algorithm
        # Continue until we've tried to connect all possible tiles
        total_tiles = self.grid_height * self.grid_width
        for iteration in range(1, total_tiles):
            # Initialize variables to track the minimum weight edge for this iteration
            min_weight = np.inf      # Current minimum weight found
            next_i, next_j = -1, -1  # Position of tile to add next
            from_i, from_j = -1, -1  # Position of MST tile that connects to next tile
            direction = None         # Direction of connection ('north', 'south', 'west', 'east')
            
            # Find all tiles currently connected to the MST
            connected_i, connected_j = np.where(tiling_indicator)
            
            # For each connected tile, examine all its unconnected neighbors
            # to find the minimum weight edge leaving the current MST
            for idx in range(len(connected_i)):
                i, j = connected_i[idx], connected_j[idx]  # Current MST tile position
                
                # Check neighbor below (north direction in image coordinates)
                # This corresponds to moving from tile (i,j) to tile (i+1,j)
                if (i + 1 < self.grid_height and                    # Within grid bounds
                    not tiling_indicator[i + 1, j] and             # Not already in MST
                    not np.isnan(self.weight_north[i + 1, j]) and   # Valid connection exists
                    self.weight_north[i + 1, j] < min_weight):     # Better than current minimum
                    
                    min_weight = self.weight_north[i + 1, j]
                    next_i, next_j = i + 1, j    # Tile to add
                    from_i, from_j = i, j        # MST tile it connects to
                    direction = 'north'
                
                # Check neighbor above (south direction in image coordinates)
                # This corresponds to moving from tile (i,j) to tile (i-1,j)
                if (i > 0 and                                      # Within grid bounds
                    not tiling_indicator[i - 1, j] and            # Not already in MST
                    not np.isnan(self.weight_north[i, j]) and      # Valid connection exists
                    self.weight_north[i, j] < min_weight):        # Better than current minimum
                    
                    min_weight = self.weight_north[i, j]
                    next_i, next_j = i - 1, j    # Tile to add
                    from_i, from_j = i, j        # MST tile it connects to
                    direction = 'south'
                
                # Check neighbor to the right (west direction in image coordinates)
                # This corresponds to moving from tile (i,j) to tile (i,j+1)
                if (j + 1 < self.grid_width and                   # Within grid bounds
                    not tiling_indicator[i, j + 1] and            # Not already in MST
                    not np.isnan(self.weight_west[i, j + 1]) and   # Valid connection exists
                    self.weight_west[i, j + 1] < min_weight):     # Better than current minimum
                    
                    min_weight = self.weight_west[i, j + 1]
                    next_i, next_j = i, j + 1    # Tile to add
                    from_i, from_j = i, j        # MST tile it connects to
                    direction = 'west'
                
                # Check neighbor to the left (east direction in image coordinates)
                # This corresponds to moving from tile (i,j) to tile (i,j-1)
                if (j > 0 and                                     # Within grid bounds
                    not tiling_indicator[i, j - 1] and           # Not already in MST
                    not np.isnan(self.weight_west[i, j]) and      # Valid connection exists
                    self.weight_west[i, j] < min_weight):        # Better than current minimum
                    
                    min_weight = self.weight_west[i, j]
                    next_i, next_j = i, j - 1    # Tile to add
                    from_i, from_j = i, j        # MST tile it connects to
                    direction = 'east'
            
                        
            # If no valid neighbor found, all remaining tiles are disconnected
            if next_i == -1:
                break
            
            # Add the selected tile to the MST
            tiling_indicator[next_i, next_j] = True
            
            # Compute the global position of the newly added tile based on the connection direction
            # Position = position of source tile + translation vector for this connection
            if direction == 'north':
                # Moving from (from_i, from_j) to (from_i+1, from_j) - going down
                global_y_pos[next_i, next_j] = (global_y_pos[from_i, from_j] + 
                                              self.translations['ty_north'][next_i, next_j])
                global_x_pos[next_i, next_j] = (global_x_pos[from_i, from_j] + 
                                              self.translations['tx_north'][next_i, next_j])
                
            elif direction == 'south':
                # Moving from (from_i, from_j) to (from_i-1, from_j) - going up
                # Use negative of the north translation from the source tile
                global_y_pos[next_i, next_j] = (global_y_pos[from_i, from_j] - 
                                              self.translations['ty_north'][from_i, from_j])
                global_x_pos[next_i, next_j] = (global_x_pos[from_i, from_j] - 
                                              self.translations['tx_north'][from_i, from_j])
                
            elif direction == 'west':
                # Moving from (from_i, from_j) to (from_i, from_j+1) - going right
                global_y_pos[next_i, next_j] = (global_y_pos[from_i, from_j] + 
                                              self.translations['ty_west'][next_i, next_j])
                global_x_pos[next_i, next_j] = (global_x_pos[from_i, from_j] + 
                                              self.translations['tx_west'][next_i, next_j])
                
            elif direction == 'east':
                # Moving from (from_i, from_j) to (from_i, from_j-1) - going left
                # Use negative of the west translation from the source tile
                global_y_pos[next_i, next_j] = (global_y_pos[from_i, from_j] - 
                                              self.translations['ty_west'][from_i, from_j])
                global_x_pos[next_i, next_j] = (global_x_pos[from_i, from_j] - 
                                              self.translations['tx_west'][from_i, from_j])
        
        # Set positions of unconnected tiles to NaN to indicate they're not part of the stitched image
        global_y_pos[~tiling_indicator] = np.nan
        global_x_pos[~tiling_indicator] = np.nan
        
        # Normalize positions to start from (0, 0) by subtracting the minimum values
        # This ensures the final stitched image has its top-left corner at origin
        min_y = np.nanmin(global_y_pos)
        min_x = np.nanmin(global_x_pos)
        global_y_pos = np.round(global_y_pos - min_y)
        global_x_pos = np.round(global_x_pos - min_x)

        return global_y_pos, global_x_pos