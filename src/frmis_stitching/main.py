import time
import argparse
import multiprocessing as mp
from frmis_stitching.stitch import FRMIS

def main():
    parser = argparse.ArgumentParser(description='FRMIS-Stitching: A Python implementation of the FRMIS stitching algorithm.')
    
    # Required arguments
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the directory containing the images to be stitched.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the final stitched mosaic.')
    
    # Mosaic grid dimensions
    parser.add_argument('--mosaic_height', type=int, required=True, help='Number of rows in the image grid.')
    parser.add_argument('--mosaic_width', type=int, required=True, help='Number of columns in the image grid.')
    
    # Image properties
    parser.add_argument('--image_height', type=int, required=True, help='Height of each image in pixels.')
    parser.add_argument('--image_width', type=int, required=True, help='Width of each image in pixels.')
    parser.add_argument('--image_channels', type=int, default=3, help='Number of channels in the images (default: 3 for color).')
    
    # Overlap and blending
    parser.add_argument('--overlap_x', type=int, required=True, help='Overlap in pixels between adjacent columns.')
    parser.add_argument('--overlap_y', type=int, required=True, help='Overlap in pixels between adjacent rows.')
    parser.add_argument('--blend', type=str, default='linear', choices=['linear', 'overlay'], help='Blending method (linear or overlay).')
    
    args = parser.parse_args()

    # The FRMIS class expects a dictionary-like object for configuration
    config = {
        'DATASET_DIR': args.dataset_dir,
        'OUTPUT_PATH': args.output_path,
        'MOSAIC_HEIGHT': args.mosaic_height,
        'MOSAIC_WIDTH': args.mosaic_width,
        'IMAGE_HEIGHT': args.image_height,
        'IMAGE_WIDTH': args.image_width,
        'IMAGE_CHANNELS': args.image_channels,
        'OVERLAP_X': args.overlap_x,
        'OVERLAP_Y': args.overlap_y,
        'BLEND': args.blend,
    }

    start_time = time.time()
    FRMIS(config).run()
    end_time = time.time()

    print("=======================")
    print(f"Stitching completed in {end_time - start_time:.2f} seconds and saved to {config['OUTPUT_PATH']}.")
    print("=======================")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main()
