import time
import os
from frmis_stitching.stitch import FRMIS

def run_benchmark():
    dataset_dir = "data\\stitch-sardana-labs-20x-5x5"
    output_path = "data\\mosaic-sardana-labs-20x-5x5.png"
    
    # Ensure dataset exists
    if not os.path.exists(dataset_dir):
        print("Dataset not found. Please run generate_dataset.py first.")
        return

    # Configuration matching the generator defaults
    config = {
        'DATASET_DIR': dataset_dir,
        'OUTPUT_PATH': output_path,
        'MOSAIC_HEIGHT': 5,
        'MOSAIC_WIDTH': 5,
        'IMAGE_HEIGHT': 3040,
        'IMAGE_WIDTH': 4056,
        'IMAGE_CHANNELS': 3,
        'OVERLAP_X': 1500,
        'OVERLAP_Y': 1500,
        'BLEND': 'overlay',
    }

    print("Starting benchmark...")
    start_time = time.time()
    FRMIS(config).run()
    end_time = time.time()
    
    print(f"Total execution time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    run_benchmark()
