import os
import cv2
import numpy as np
import argparse

def generate_synthetic_dataset(output_dir, rows, cols, img_h, img_w, overlap_x, overlap_y):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a large base image
    full_h = rows * img_h - (rows - 1) * overlap_y
    full_w = cols * img_w - (cols - 1) * overlap_x
    
    # Create a pattern (checkerboard or random noise) to ensure features for SIFT
    base_image = np.zeros((full_h, full_w), dtype=np.uint8)
    
    # Add random noise
    noise = np.random.randint(0, 256, (full_h, full_w), dtype=np.uint8)
    base_image = cv2.addWeighted(base_image, 0.5, noise, 0.5, 0)
    
    # Add some shapes/lines to ensure good features
    for _ in range(100):
        pt1 = (np.random.randint(0, full_w), np.random.randint(0, full_h))
        pt2 = (np.random.randint(0, full_w), np.random.randint(0, full_h))
        cv2.line(base_image, pt1, pt2, (255), 2)
        
    for _ in range(100):
        center = (np.random.randint(0, full_w), np.random.randint(0, full_h))
        radius = np.random.randint(10, 50)
        cv2.circle(base_image, center, radius, (255), -1)

    print(f"Generated base image of size {full_w}x{full_h}")

    # Slice the image into tiles
    count = 0
    for i in range(rows):
        for j in range(cols):
            y_start = i * (img_h - overlap_y)
            x_start = j * (img_w - overlap_x)
            
            tile = base_image[y_start:y_start+img_h, x_start:x_start+img_w]
            
            # Save tile
            filename = f"img_{count:04d}.png"
            cv2.imwrite(os.path.join(output_dir, filename), tile)
            count += 1
            
    print(f"Saved {count} images to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument("--width", type=int, default=500)
    parser.add_argument("--overlap_x", type=int, default=50)
    parser.add_argument("--overlap_y", type=int, default=50)
    args = parser.parse_args()
    
    generate_synthetic_dataset(args.output_dir, args.rows, args.cols, args.height, args.width, args.overlap_x, args.overlap_y)
