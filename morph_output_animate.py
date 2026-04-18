import os
import argparse
import sys
from PIL import Image
import imageio.v2 as imageio

# 1. Setup Arguments
parser = argparse.ArgumentParser(description="Generate a rollout video.")
parser.add_argument("results_dir", type=str, help="Directory containing images")
parser.add_argument("prefix", type=str, help="Filename prefix before the number")
args = parser.parse_args()

results_dir = args.results_dir
prefix = args.prefix
out_path = "rollout_video.mp4"

# 2. Load Images
images = []
for t in range(10):
    path = os.path.join(results_dir, f"{prefix}{t}.png")
    if not os.path.exists(path):
        # Check for .jpg if .png fails since your upload was a .jpg
        path = path.replace(".png", ".jpg")
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            sys.exit(1)
    images.append(Image.open(path))

# 3. Crop and Tile
img_w, img_h = images[0].size
col_w = img_w // 5
row_h = img_h // 2  # We'll use Ch0 (top row)

target_col = 1
pred_col = 2
mse_col = 3

frames = []
for t in range(10):
    img = images[t]
    
    # Crop specific tiles from the grid (Ch0)
    target_crop = img.crop((target_col * col_w, 0, (target_col + 1) * col_w, row_h))
    pred_crop   = img.crop((pred_col * col_w,   0, (pred_col + 1) * col_w,   row_h))
    mse_crop    = img.crop((mse_col * col_w,    0, (mse_col + 1) * col_w,    row_h))

    # Create horizontal layout: [ TARGET | PREDICTION | MSE ]
    frame = Image.new("RGB", (col_w * 3, row_h))
    frame.paste(target_crop, (0, 0))
    frame.paste(pred_crop,   (col_w, 0))
    frame.paste(mse_crop,    (2 * col_w, 0))

    frames.append(frame)

# 4. Save Video
imageio.mimsave(out_path, frames, fps=2)
print(f"Success! Video saved to: {os.path.abspath(out_path)}")