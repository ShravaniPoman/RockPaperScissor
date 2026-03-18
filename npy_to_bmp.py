# npy_to_bmp.py — Runs on your LAPTOP
# Converts all .npy training images to viewable BMP files
# Creates rock_bmp/, paper_bmp/, scissors_bmp/ folders
#
# Usage: python3 npy_to_bmp.py

import os
import struct
import numpy as np

DATA_DIR = "training_data_v2"
CLASSES = ["rock", "paper", "scissors"]

def save_as_bmp(pixels, filepath):
    """Convert a 32x32 float array (0.0-1.0) to an 8-bit grayscale BMP file."""
    W, H = 32, 32
    HEADER_SIZE = 14
    DIB_SIZE = 40
    PALETTE_SIZE = 256 * 4
    ROW_PADDING = W % 4
    PIXEL_DATA_SIZE = (W + ROW_PADDING) * H
    FILE_SIZE = HEADER_SIZE + DIB_SIZE + PALETTE_SIZE + PIXEL_DATA_SIZE

    bmp = bytearray(FILE_SIZE)

    # BMP file header
    bmp[0:2] = b'BM'
    struct.pack_into('<I', bmp, 2, FILE_SIZE)
    struct.pack_into('<I', bmp, 10, HEADER_SIZE + DIB_SIZE + PALETTE_SIZE)

    # DIB header
    struct.pack_into('<I', bmp, 14, DIB_SIZE)
    struct.pack_into('<i', bmp, 18, W)
    struct.pack_into('<i', bmp, 22, H)
    struct.pack_into('<H', bmp, 26, 1)
    struct.pack_into('<H', bmp, 28, 8)
    struct.pack_into('<I', bmp, 34, PIXEL_DATA_SIZE)

    # Grayscale palette
    for i in range(256):
        off = HEADER_SIZE + DIB_SIZE + i * 4
        bmp[off] = i
        bmp[off+1] = i
        bmp[off+2] = i
        bmp[off+3] = 0

    # Pixel data (BMP is bottom-up)
    pixel_offset = HEADER_SIZE + DIB_SIZE + PALETTE_SIZE
    for y in range(H):
        for x in range(W):
            val = int(pixels[H - 1 - y][x] * 255)
            val = max(0, min(255, val))
            bmp[pixel_offset] = val
            pixel_offset += 1
        pixel_offset += ROW_PADDING

    with open(filepath, 'wb') as f:
        f.write(bmp)

def main():
    total = 0
    for cls in CLASSES:
        npy_folder = os.path.join(DATA_DIR, cls)
        bmp_folder = os.path.join(DATA_DIR, cls + "_bmp")

        if not os.path.exists(npy_folder):
            print(f"Folder not found: {npy_folder}")
            continue

        os.makedirs(bmp_folder, exist_ok=True)
        files = sorted([f for f in os.listdir(npy_folder) if f.endswith('.npy')])
        print(f"Converting {cls}: {len(files)} images")

        for f in files:
            pixels = np.load(os.path.join(npy_folder, f))
            if pixels.shape == (32, 32):
                bmp_path = os.path.join(bmp_folder, f.replace('.npy', '.bmp'))
                save_as_bmp(pixels, bmp_path)
                total += 1

    print(f"\nDone! Converted {total} images.")
    for cls in CLASSES:
        bmp_folder = os.path.join(DATA_DIR, cls + "_bmp")
        if os.path.exists(bmp_folder):
            n = len([f for f in os.listdir(bmp_folder) if f.endswith('.bmp')])
            print(f"  {bmp_folder}/ ({n} images)")

if __name__ == "__main__":
    main()
