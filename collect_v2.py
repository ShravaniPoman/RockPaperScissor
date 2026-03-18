# collect_v2.py — Runs on your LAPTOP
# Purpose: Collects training data for the rock/paper/scissors CNN classifier.
#          Captures images from the ESP32 camera over WiFi and processes them
#          through the EXACT same pipeline as classify_laptop.py.
#
# CRITICAL DESIGN DECISION: Training data must be processed identically to
# inference data. This script uses the same resize, brightness, contrast, and
# normalization as classify_laptop.py. Without this match, the model achieves
# high validation accuracy but fails on live images.
#
# Images are saved as NumPy .npy arrays (32x32 float32, values 0.0-1.0)
# organized in folders: training_data_v2/rock/, paper/, scissors/
#
# Usage:
#   1. Run socket_server.py on ESP32 first
#   2. python3 collect_v2.py
#   3. Select r/p/s, then ENTER for single capture or number for burst
#
# Author: Shravani Poman (with assistance from Claude AI)


import socket
import struct
import os
import time
import numpy as np

# --- Connection Settings ---
ESP32_IP = "172.20.10.4"  # IP address of ESP32 — update if it changes
ESP32_PORT = 8080          # Must match PORT in socket_server.py

# --- Data Organization ---
BASE_DIR = "training_data_v2"           # Root folder for all training data
CLASSES = ["rock", "paper", "scissors"] # The three gesture classes to classify

# --- Image Enhancement Settings ---
# These MUST match the values in classify_laptop.py for pipeline consistency
BRIGHTNESS_BOOST = 30   # Added to each pixel value (compensates for dark camera images)
CONTRAST_FACTOR = 1.5   # Multiplied around midpoint 128 (increases distinction between light/dark)

# Track the next available file number for each class to prevent overwriting
counters = {"rock": 0, "paper": 0, "scissors": 0}

def setup():
    """Create output folders and initialize file counters from existing files.
    Scans existing .npy files to find the highest number, so new captures
    continue numbering from where we left off (even across sessions)."""
    for cls in CLASSES:
        path = os.path.join(BASE_DIR, cls)
        os.makedirs(path, exist_ok=True)  # Create folder if it doesn't exist
        files = [f for f in os.listdir(path) if f.endswith('.npy')]
        if files:
            # Find the highest existing file number to continue from
            nums = []
            for f in files:
                try:
                    nums.append(int(f.replace(cls + "_", "").replace(".npy", "")))
                except:
                    pass
            counters[cls] = max(nums) + 1 if nums else 0

def receive_image(sock):
    """Receive a BMP image from the ESP32 socket server.
    Protocol: First 4 bytes = image size (big-endian), then raw BMP data.
    Reads in chunks of up to 4096 bytes until full image is received."""
    size_data = sock.recv(4)  # Read 4-byte size header
    if len(size_data) < 4:
        return None
    img_size = int.from_bytes(size_data, 'big')  # Decode big-endian integer
    # Read image data in chunks until we have all bytes
    img_data = b""
    while len(img_data) < img_size:
        chunk = sock.recv(min(4096, img_size - len(img_data)))
        if not chunk:
            break
        img_data += chunk
    return img_data

def process_image(bmp_data):
    """Process a raw BMP image into a 32x32 normalized float array.
    This function implements the EXACT same processing as classify_laptop.py's
    resize_bmp_to_32x32() to ensure training/inference pipeline match.
    
    Steps:
    1. Parse BMP header to get image dimensions and pixel data offset
    2. Read raw pixel values into a 2D array
    3. Resize from 128x128 to 32x32 using nearest-neighbor interpolation
    4. Apply brightness boost (+30) and contrast enhancement (x1.5)
    5. Normalize to 0.0-1.0 float range
    
    Returns: 32x32 NumPy float32 array, or None if invalid BMP"""
    
    # Validate BMP magic bytes
    if bmp_data[0:2] != b'BM':
        return None

    # Parse BMP header fields using struct (little-endian format)
    data_offset = struct.unpack_from('<I', bmp_data, 10)[0]   # Byte offset to pixel data
    width = struct.unpack_from('<i', bmp_data, 18)[0]          # Image width in pixels
    height = struct.unpack_from('<i', bmp_data, 22)[0]         # Image height (negative = top-down)
    bits_per_pixel = struct.unpack_from('<H', bmp_data, 28)[0] # Bits per pixel (8 for grayscale)

    abs_height = abs(height)
    # BMP rows are padded to 4-byte boundaries
    row_size = ((width * bits_per_pixel + 31) // 32) * 4

    # Read all pixels from BMP into a 2D list
    pixels = []
    for y in range(abs_height):
        row = []
        for x in range(width):
            offset = data_offset + y * row_size + x * (bits_per_pixel // 8)
            if offset < len(bmp_data):
                pixel = bmp_data[offset] & 0xFF  # Ensure unsigned 0-255
            else:
                pixel = 0
            row.append(pixel)
        pixels.append(row)

    # Resize to 32x32 using nearest-neighbor interpolation
    # Maps each output pixel to the closest input pixel
    resized = np.zeros((32, 32), dtype=np.float32)
    for new_y in range(32):
        old_y = (new_y * abs_height) // 32  # Map new Y to original Y
        for new_x in range(32):
            old_x = (new_x * width) // 32   # Map new X to original X
            if old_y < len(pixels) and old_x < len(pixels[old_y]):
                val = pixels[old_y][old_x]
            else:
                val = 0
            # Apply brightness boost (compensates for dark camera)
            val = val + BRIGHTNESS_BOOST
            # Apply contrast enhancement around midpoint (128)
            # Formula: new = (old - 128) * factor + 128
            val = int((val - 128) * CONTRAST_FACTOR + 128)
            # Clamp to valid pixel range
            val = max(0, min(255, val))
            # Normalize to 0.0-1.0 for neural network input
            resized[new_y][new_x] = val / 255.0

    return resized

def capture_fresh(sock):
    """Capture a fresh image by discarding stale frames from camera buffer.
    The ESP32 camera maintains a frame buffer, so the first capture() may
    return an old cached image. We discard 2 frames before taking the real one."""
    for _ in range(2):
        sock.send(b"capture")
        receive_image(sock)     # Receive and discard stale frame
        time.sleep(0.1)         # Brief delay between captures
    sock.send(b"capture")
    return receive_image(sock)  # This frame is fresh

def show_status():
    """Display current image count for each class with a visual bar chart."""
    print("\n========================================")
    for cls in CLASSES:
        path = os.path.join(BASE_DIR, cls)
        n = len([f for f in os.listdir(path) if f.endswith('.npy')]) if os.path.exists(path) else 0
        bar = "#" * min(n // 5, 30)  # Each # represents 5 images
        print(f"  {cls:10s}: {n:4d}  {bar}")
    print("========================================")

def main():
    """Main data collection loop. Connects to ESP32 and provides interactive
    menu for capturing rock/paper/scissors images."""
    setup()  # Create folders and initialize counters
    print("Connecting to ESP32...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        sock.connect((ESP32_IP, ESP32_PORT))
        print("Connected!")

        while True:
            show_status()
            print("\nCommands: r/p/s = capture, q = quit")
            choice = input("Choice: ").strip().lower()

            if choice == 'q':
                sock.send(b"quit")  # Tell server we're disconnecting
                break
            elif choice in ['r', 'p', 's']:
                # Map shortcut to full class name
                cls = {"r": "rock", "p": "paper", "s": "scissors"}[choice]
                print(f"\nCapturing [{cls.upper()}]")
                print("  ENTER = capture one, number = burst, done = go back\n")

                while True:
                    cmd = input(f"  [{cls}] > ").strip().lower()
                    if cmd == 'done':
                        break  # Return to class selection menu
                    elif cmd == '':
                        # Single capture mode - one image per Enter press
                        img_data = capture_fresh(sock)
                        if img_data:
                            pixels = process_image(img_data)
                            if pixels is not None:
                                num = counters[cls]
                                counters[cls] += 1  # Increment for next file
                                filepath = os.path.join(BASE_DIR, cls, f"{cls}_{num:04d}.npy")
                                np.save(filepath, pixels)  # Save as NumPy array
                                total = len([f for f in os.listdir(os.path.join(BASE_DIR, cls)) if f.endswith('.npy')])
                                print(f"    Saved! ({total} total {cls})")
                    elif cmd.isdigit():
                        # Burst capture mode - capture N images rapidly
                        num = int(cmd)
                        print(f"    Burst: {num} images. Hold steady!")
                        time.sleep(0.5)  # Give user time to position hand
                        for i in range(num):
                            img_data = capture_fresh(sock)
                            if img_data:
                                pixels = process_image(img_data)
                                if pixels is not None:
                                    n = counters[cls]
                                    counters[cls] += 1
                                    filepath = os.path.join(BASE_DIR, cls, f"{cls}_{n:04d}.npy")
                                    np.save(filepath, pixels)
                                    total = len([f for f in os.listdir(os.path.join(BASE_DIR, cls)) if f.endswith('.npy')])
                                    print(f"    [{i+1}/{num}] Saved ({total} total)")
                            time.sleep(0.1)  # Brief pause between burst captures
                        print(f"    Done!")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()
        show_status()  # Show final counts

if __name__ == "__main__":
    main()
