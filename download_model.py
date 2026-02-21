#!/usr/bin/env python3
"""
Download SSD MobileNet V1 TFLite model and COCO label map for SmartFocus AI.
Run this script once before building the app.
Usage: python download_model.py
"""

import urllib.request
import zipfile
import os
import shutil

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "app", "src", "main", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

MODEL_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
ZIP_PATH = os.path.join(ASSETS_DIR, "model.zip")

print("Downloading SSD MobileNet V1 TFLite model...")
urllib.request.urlretrieve(MODEL_URL, ZIP_PATH)
print("Download complete.")

print("Extracting model files...")
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    z.extractall(ASSETS_DIR)

# Rename to standard names expected by the app
for f in os.listdir(ASSETS_DIR):
    if f.endswith(".tflite"):
        src = os.path.join(ASSETS_DIR, f)
        dst = os.path.join(ASSETS_DIR, "ssd_mobilenet_v1.tflite")
        shutil.move(src, dst)
        print(f"Model saved as: {dst}")
    elif "label" in f.lower() and f.endswith(".txt"):
        src = os.path.join(ASSETS_DIR, f)
        dst = os.path.join(ASSETS_DIR, "labelmap.txt")
        shutil.move(src, dst)
        print(f"Labels saved as: {dst}")

os.remove(ZIP_PATH)
print("Done! Model and labels are ready in the assets/ folder.")
