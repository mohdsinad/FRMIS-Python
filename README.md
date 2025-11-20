# FRMIS-Stitching: Fast Robust Multi-Image Stitching

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A command-line tool for stitching a grid of images into a single, seamless mosaic. This project is a Python implementation of the Fast Robust Multi-Image Stitching (FRMIS) algorithm, designed for speed and accuracy in creating large-scale panoramas from tiled image collections.

---

## Features

- **Robust Pairwise Alignment**: Uses feature detection (SIFT) and RANSAC to accurately calculate translations between adjacent images.
- **Global Optimization**: Employs a Minimum Spanning Tree (MST) approach on a graph of image connections to find the optimal global position for each tile, minimizing accumulated error.
- **Blending Options**: Supports both simple overlay and weighted linear blending to create seamless transitions between images.
- **Command-Line Interface**: Easy-to-use CLI for running the stitching process with configurable parameters.
- **Grid-Based**: Optimized for image sets captured in a grid-like pattern (e.g., microscopy, aerial photography).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mohdsinad/FRMIS-Python.git
    cd FRMIS-Python
    ```

2.  **Install dependencies and the package:**
    It is recommended to use a virtual environment.
    ```bash
    # Create and activate a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate

    # Install the package
    pip install .
    ```
    This will install the necessary libraries from `requirements.txt` and make the `frmis-stitch` command available in your terminal.

## Usage

The tool is run from the command line. You must provide parameters specifying the input image directory, output path, and image grid dimensions.

### Command-Line Arguments

| Argument              | Shorthand | Description                                                               | Required |
| --------------------- | --------- | ------------------------------------------------------------------------- | :------: |
| `--dataset_dir`       | `-d`      | Path to the directory containing the input images.                        |   Yes    |
| `--output_path`       | `-o`      | Path to save the final stitched mosaic image.                             |   Yes    |
| `--mosaic_height`     | `-mh`     | The number of rows in the image grid.                                     |   Yes    |
| `--mosaic_width`      | `-mw`     | The number of columns in the image grid.                                  |   Yes    |
| `--image_height`      | `-ih`     | The height of a single image in pixels.                                   |   Yes    |
| `--image_width`       | `-iw`     | The width of a single image in pixels.                                    |   Yes    |
| `--overlap_x`         | `-ox`     | The approximate horizontal overlap between adjacent images in pixels.     |   Yes    |
| `--overlap_y`         | `-oy`     | The approximate vertical overlap between adjacent images in pixels.       |   Yes    |
| `--image_channels`    | `-ic`     | The number of channels in the images (e.g., 3 for RGB). Default: `3`.     |    No    |
| `--blend`             | `-b`      | The blending method to use (`overlay` or `linear`). Default: `linear`.    |    No    |

### Example

To run the stitching process on a 4x5 grid of 1024x1024 images:

```bash
frmis-stitch \
    --dataset_dir /path/to/your/images \
    --output_path /path/to/output/final_mosaic.png \
    --mosaic_height 4 \
    --mosaic_width 5 \
    --image_height 1024 \
    --image_width 1024 \
    --overlap_x 150 \
    --overlap_y 150 \
    --blend linear
```

### Input Data
The tool expects the input images in `dataset_dir` to be named in a way that allows for natural sorting (e.g., `image_01.tif`, `image_02.tif`, ...). The images will be arranged into the grid row by row.

## Algorithm Overview

The FRMIS algorithm operates in three main stages:

1.  **Pairwise Alignment**: The algorithm first considers every pair of adjacent images (horizontally and vertically). It identifies matching feature points in their overlapping regions and calculates the precise `(tx, ty)` translation required to align them. The number of matching points (inliers) is used as a confidence score for that alignment.

2.  **Global Alignment (MST Optimization)**: A simple chain of translations can quickly accumulate errors. To solve this, the algorithm builds a graph where each image is a node and each potential alignment is an edge, weighted by its confidence score. A Minimum Spanning Tree (MST) is then constructed to find the single best path connecting all images, ensuring a globally consistent layout with the least possible error.

3.  **Blending and Rendering**: Once the final position of each tile is determined, the images are placed onto a large canvas. A blending function (e.g., linear feathering) is applied at the seams to smooth the transition between images, resulting in a seamless final mosaic.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
