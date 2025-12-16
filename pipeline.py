import os
import time
import tempfile
import uuid
import sys
import glob 
import subprocess # <-- NEW: Added for calling vips.exe directly

# ====================================================================
# === VIPS Helper Function (Finds the path for portability) ===
# ====================================================================

def find_vips_path_on_windows(initial_path):
    """
    Attempts to find the VIPS bin directory containing the necessary DLLs.
    This function ensures the script works without hardcoding the path every time.
    """
    
    # 1. Check the initial path provided by the user (most likely location)
    if os.path.isdir(initial_path) and os.path.exists(os.path.join(initial_path, 'vips.exe')):
        print(f"Found VIPS bin in initial path: {initial_path}")
        return initial_path

    # 2. Search common VIPS installation folders on the C drive
    print("Initial VIPS path invalid. Searching common locations...")
    
    # Check C:\ directly for standard vips-dev-* folders
    for base_dir in ['C:\\']:
        # This searches for C:\vips-dev-8.17\bin, C:\vips-dev-8.16\bin, etc.
        search_pattern = os.path.join(base_dir, 'vips-dev-*', 'bin')
        found_paths = glob.glob(search_pattern)
        
        # Sort to pick the highest version path if multiple exist
        found_paths.sort(reverse=True)
        
        if found_paths:
            found_path = found_paths[0]
            if os.path.exists(os.path.join(found_path, 'vips.exe')):
                print(f"Found VIPS bin automatically at: {found_path}")
                return found_path
                
    # 3. If still not found, return None
    return None

# ====================================================================
# === Configuration ===
# ====================================================================

# 1. Paths (Only update this if auto-detection fails!)
DATASET_DIR = r"C:\Users\mimd\Pictures\RAW\cervical_pid_02"
FINAL_OUTPUT_PATH = r"C:\Users\mimd\Pictures\WSI\mosaic-cervical-pid-02-vips.tiff"
# VIPS_BIN_PATH is the starting point for the automated search
VIPS_BIN_PATH = r"C:\Users\mimd\vips-dev-8.17\bin" 

# 2. Stitching Parameters
STITCH_CONFIG = {
    'MOSAIC_HEIGHT': 10, 'MOSAIC_WIDTH': 10,
    'IMAGE_HEIGHT': 3040, 'IMAGE_WIDTH': 4056,
    'IMAGE_CHANNELS': 3, 'OVERLAP_X': 1500, 'OVERLAP_Y': 1500,
    'BLEND': 'linear',
}

# 3. VIPS Compression Parameters
# We are now using JPEG compression, as ZSTD was causing issues.
JP2_QUALITY = 90
TILE_SIZE = 512

# ====================================================================
# === VIPS Setup (CRITICAL FOR WINDOWS) ===
# ====================================================================

# Attempt to locate VIPS bin path automatically
VIPS_BIN_PATH = find_vips_path_on_windows(VIPS_BIN_PATH)

if VIPS_BIN_PATH is None:
    print("\n" + "="*50)
    print("❌ FATAL ERROR: VIPS bin directory could not be located.")
    print("Please check your VIPS installation or manually update:")
    print("the VIPS_BIN_PATH variable near the top of the script.")
    print("="*50)
    sys.exit(1)

# === USING pyvips DOCUMENTATION METHOD ===
# This tells Python where to find the VIPS DLLs before the import
# We use os.pathsep (which is ';' on Windows) for robustness
os.environ['PATH'] = VIPS_BIN_PATH + os.pathsep + os.environ['PATH']

# Import pyvips AFTER the path has been set (pyvips is still required for its dependencies)
try:
    import pyvips
    from frmis_stitching.stitch import FRMIS
except ImportError as e:
    print(f"\n❌ Error: Failed to import a required module: {e}")
    print("Ensure you have run 'pip install pyvips' and installed the FRMIS package.")
    sys.exit(1)


# ====================================================================
# === Main Execution ===
# ====================================================================

def run_pipeline():
    # Generate a unique temporary file path for the stitched output
    temp_dir = tempfile.gettempdir()
    temp_filename = f"stitching_intermediate_{uuid.uuid4().hex}.tiff"
    intermediate_path = os.path.join(temp_dir, temp_filename)

    print(f"\n--- Pipeline Started ---")
    print(f"Temporary Stitched File will be at: {intermediate_path}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(FINAL_OUTPUT_PATH), exist_ok=True)

    pipeline_start = time.time()

    try:
        # --- Phase 1: Stitching Images (Save to Temp) ---
        print("\n--- Phase 1: Stitching Images (FRMIS) ---")
        
        stitch_run_config = STITCH_CONFIG.copy()
        stitch_run_config['DATASET_DIR'] = DATASET_DIR
        stitch_run_config['OUTPUT_PATH'] = intermediate_path
        
        stitch_start = time.time()
        FRMIS(stitch_run_config).run()
        print(f"✅ Stitching complete in {time.time() - stitch_start:.2f}s")

        if not os.path.exists(intermediate_path):
            raise FileNotFoundError("Intermediate stitching file was not created.")

        # --- Phase 2: VIPS Optimization (Temp -> Final using Command Line) ---
        print(f"\n--- Phase 2: VIPS Optimization (vips.exe command line) ---")
        print(f"   Input: {os.path.basename(intermediate_path)}")
        print(f"   Output: {os.path.basename(FINAL_OUTPUT_PATH)}")

        vips_start = time.time()
        
        # --- Using VIPS Command Line (vips.exe) for maximum stability on Windows ---
        # This bypasses the Python DLL conflicts that caused the ZSTD error.
        try:
            subprocess.run([
                os.path.join(VIPS_BIN_PATH, 'vips.exe'), 
                "tiffsave",
                intermediate_path,                    # Input: Stitched file
                FINAL_OUTPUT_PATH,                    # Output: Final WSI file
                
                # WSI Structure Parameters
                "--tile",
                "--tile-width", str(TILE_SIZE),
                "--tile-height", str(TILE_SIZE),
                "--pyramid",
                "--bigtiff", 
                
                # DEFLATE COMPRESSION PARAMETERS
                "--compression", "lzw",
                "--predictor", "horizontal"
                
            ], check=True, capture_output=True, text=True)
            
        except subprocess.CalledProcessError as e:
            # Print the error output from vips.exe for debugging
            print(f"   ❌ VIPS optimization failed via command-line. Error:")
            print(f"      Stdout: {e.stdout}")
            print(f"      Stderr: {e.stderr}")
            # Re-raise the exception to be caught by the outer block
            raise Exception("VIPS command-line optimization failed.") from e
        
        print(f"✅ Optimization complete in {time.time() - vips_start:.2f}s")

        # Stats
        orig_size = os.path.getsize(intermediate_path) / (1024 * 1024)
        final_size = os.path.getsize(FINAL_OUTPUT_PATH) / (1024 * 1024)
        print(f"   Original Size (stitched): {orig_size:.2f} MB")
        print(f"   Final Size (optimized):   {final_size:.2f} MB")

    except Exception as e:
        print(f"❌ An error occurred during the pipeline: {e}")

    finally:
        # --- Cleanup ---
        print("\n--- Cleanup ---")
        if os.path.exists(intermediate_path):
            try:
                os.remove(intermediate_path)
                print(f"🗑️  Temporary stitched file deleted.")
            except OSError as e:
                print(f"⚠️  Warning: Could not delete temp file: {e}")
        
    print(f"\nTotal Pipeline Time: {time.time() - pipeline_start:.2f}s")

if __name__ == "__main__":
    run_pipeline()