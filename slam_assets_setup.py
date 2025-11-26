import os
import urllib.request
import tarfile

def create_config_yaml():
    """Generates a standard config file for a 640x480 webcam."""
    content = """%YAML:1.0

#--------------------------------------------------------------------------------------------
# Camera Parameters. Adjust them!
#--------------------------------------------------------------------------------------------
File.version: "1.0"

Camera.type: "PinHole"

# Camera calibration and distortion parameters (OpenCV) 
# These are standard values for a generic 640x480 webcam.
# For high precision, you should calibrate your specific camera using OpenCV.
Camera.fx: 500.0
Camera.fy: 500.0
Camera.cx: 320.0
Camera.cy: 240.0

# Distortion coefficients (k1, k2, p1, p2)
# Assume 0 for generic webcam (no fisheye)
Camera.k1: 0.0
Camera.k2: 0.0
Camera.p1: 0.0
Camera.p2: 0.0

Camera.width: 640
Camera.height: 480

# Camera frames per second 
Camera.fps: 30.0

# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
Camera.RGB: 1

#--------------------------------------------------------------------------------------------
# ORB Parameters
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: 1000

# ORB Extractor: Scale factor between levels in the scale pyramid 	
ORBextractor.scaleFactor: 1.2

# ORB Extractor: Number of levels in the scale pyramid	
ORBextractor.nLevels: 8

# ORB Extractor: Fast threshold
# Image is divided in a grid. At each cell FAST are extracted imposing a minimum response.
# Firstly we impose iniThFAST. If no corners are detected we impose a lower value minThFAST
# You can lower these values if your images have low contrast			
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2
Viewer.CameraLineWidth: 3
Viewer.ViewpointX: 0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500
"""
    with open("webcam_config.yaml", "w") as f:
        f.write(content)
    print("✅ Created 'webcam_config.yaml'")


def download_vocab():
    """Downloads and extracts the ORB Vocabulary."""
    url = "https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/master/Vocabulary/ORBvoc.txt.tar.gz"
    tar_name = "ORBvoc.txt.tar.gz"
    final_name = "ORBvoc.txt"

    if os.path.exists(final_name):
        print(f"✅ '{final_name}' already exists.")
        return

    print(f"⬇️  Downloading {tar_name} (approx 150MB compressed)...")
    try:
        urllib.request.urlretrieve(url, tar_name)
        print("📦 Extracting...")
        with tarfile.open(tar_name, "r:gz") as tar:
            tar.extractall()

        # Cleanup
        if os.path.exists(tar_name):
            os.remove(tar_name)

        print(f"✅ Success! '{final_name}' is ready.")

    except Exception as e:
        print(f"❌ Error downloading vocabulary: {e}")
        print("👉 Please download manually from: https://github.com/UZ-SLAMLab/ORB_SLAM3/tree/master/Vocabulary")


if __name__ == "__main__":
    print("--- Setting up ORB-SLAM3 Assets ---")
    create_config_yaml()
    download_vocab()
    print("\nSetup complete. You can now run your SLAM script.")