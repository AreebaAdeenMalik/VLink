import torch
import os
import shutil


def clean_cache():
    # Get the default PyTorch Hub cache directory
    # On Windows this is usually: C:\Users\<User>\.cache\torch\hub
    hub_dir = torch.hub.get_dir()

    # The specific folder where model weights (like ResNext) are stored
    checkpoints_dir = os.path.join(hub_dir, 'checkpoints')

    print(f"Checking for checkpoints in: {checkpoints_dir}")

    if os.path.exists(checkpoints_dir):
        try:
            print("Found checkpoint directory. Deleting to force re-download...")
            shutil.rmtree(checkpoints_dir)
            print("✅ Success! Corrupted weights deleted.")
            print("Now run your 'midas_inference.py' script again to download fresh weights.")
        except Exception as e:
            print(f"❌ Error deleting folder: {e}")
            print(f"Please manually delete the folder: {checkpoints_dir}")
    else:
        print("No checkpoints directory found. Nothing to clean.")


if __name__ == "__main__":
    clean_cache()