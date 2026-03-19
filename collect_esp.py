# collect_esp.py — Runs on your LAPTOP
#
# Purpose: Collects training data for ESP32 on-device classification.
#          Receives raw grayscale BMP images from the ESP32 camera via WiFi,
#          resizes to 32x32, normalizes to 0.0-1.0, and saves as NumPy arrays.
#
# CRITICAL: Processing must match classify.py's resize_and_extract() EXACTLY.
#           Both handle 8-bit grayscale BMP, negative height (top-down),
#           nearest-neighbor resize, and normalize to 0.0-1.0.
#           Any mismatch causes the model to fail on live images.
#
# Usage:
#   1. Run socket_server.py on ESP32 (with WiFi connected)
#   2. python3 collect_esp.py
#   3. Select r/p/s, then ENTER for single or number for burst capture
#
# Author: Shravani Poman (with assistance from Claude AI)
# Course: AI for Engineers, Spring 2026

import socket
import struct
import os
import time
import numpy as np

# --- Connection Settings ---
ESP32_IP = "172.20.10.4"  # ESP32's IP address — update if it changes
ESP32_PORT = 8080          # Must match socket_server.py PORT

# --- Data Organization ---
BASE_DIR = "training_data_esp"           # Root folder for training data
CLASSES = ["rock", "paper", "scissors"]  # Class labels

# File number counters — tracks next available number per class
counters = {"rock": 0, "paper": 0, "scissors": 0}

def setup():
    """Create output folders and initialize counters from existing files.
    Scans existing .npy files to continue numbering across sessions."""
    for cls in CLASSES:
        path = os.path.join(BASE_DIR, cls)
        os.makedirs(path, exist_ok=True)
        files = [f for f in os.listdir(path) if f.endswith('.npy')]
        if files:
            # Find highest existing file number to continue from
            nums = []
            for f in files:
                try:
                    nums.append(int(f.replace(cls + "_", "").replace(".npy", "")))
                except:
                    pass
            counters[cls] = max(nums) + 1 if nums else 0

def receive_image(sock):
    """Receive a BMP image from ESP32 socket server.
    Protocol: 4-byte big-endian size header + raw BMP data.
    Reads in chunks until full image received."""
    size_data = sock.recv(4)  # Read 4-byte size header
    if len(size_data) < 4:
        return None
    img_size = int.from_bytes(size_data, 'big')  # Decode size
    # Accumulate image data in chunks
    img_data = b""
    while len(img_data) < img_size:
        chunk = sock.recv(min(4096, img_size - len(img_data)))
        if not chunk:
            break
        img_data += chunk
    return img_data

def process_raw_image(bmp_data):
    """Process raw grayscale BMP into 32x32 normalized array.
    MUST match classify.py's resize_and_extract() exactly.
    
    Steps:
    1. Parse BMP header (offset, width, height, bpp)
    2. Read pixels handling both 8-bit and 24-bit formats
    3. Resize 128x128 to 32x32 with nearest-neighbor
    4. Handle top-down (negative height) BMP storage
    5. Normalize to 0.0-1.0 float range
    
    Returns: 32x32 NumPy float32 array, or None if invalid"""
    if bmp_data[0:2] != b'BM':
        return None
    
    # Parse BMP header fields
    data_offset = struct.unpack_from('<I', bmp_data, 10)[0]   # Pixel data start
    width = struct.unpack_from('<i', bmp_data, 18)[0]          # Image width
    height = struct.unpack_from('<i', bmp_data, 22)[0]         # Height (negative = top-down)
    bpp = struct.unpack_from('<H', bmp_data, 28)[0]            # Bits per pixel
    
    abs_height = abs(height)
    top_down = height < 0  # OV3660 outputs top-down BMPs
    bytes_per_pixel = bpp // 8  # 1 for grayscale, 3 for color
    row_size = ((width * bpp + 31) // 32) * 4  # Row padding to 4-byte boundary
    
    # Read all pixels as grayscale values (0-255)
    raw = []
    for y in range(abs_height):
        for x in range(width):
            offset = data_offset + y * row_size + x * bytes_per_pixel
            if offset < len(bmp_data):
                if bytes_per_pixel == 3:
                    # 24-bit color: convert BGR to grayscale
                    b = bmp_data[offset]
                    g = bmp_data[offset + 1]
                    r = bmp_data[offset + 2]
                    gray = (r * 77 + g * 150 + b * 29) >> 8  # Standard conversion
                    raw.append(gray)
                else:
                    # 8-bit grayscale: direct read
                    raw.append(bmp_data[offset])
            else:
                raw.append(0)
    
    # Resize to 32x32 using nearest-neighbor interpolation
    pixels = np.zeros((32, 32), dtype=np.float32)
    for new_y in range(32):
        if top_down:
            old_y = (new_y * abs_height) // 32  # Top-down: direct mapping
        else:
            old_y = abs_height - 1 - (new_y * abs_height) // 32  # Bottom-up: flip
        
        for new_x in range(32):
            old_x = (new_x * width) // 32
            idx = old_y * width + old_x
            if idx < len(raw):
                pixels[new_y][new_x] = raw[idx] / 255.0  # Normalize to 0.0-1.0
            else:
                pixels[new_y][new_x] = 0.0
    
    return pixels

def capture_fresh(sock):
    """Capture a fresh image by discarding stale frames from camera buffer.
    The ESP32 camera buffers old frames — discard 2 before taking real one."""
    for _ in range(2):
        sock.send(b"capture")
        receive_image(sock)
        time.sleep(0.1)
    sock.send(b"capture")
    return receive_image(sock)

def show_status():
    """Display current image count per class with visual bar."""
    print("\n========================================")
    for cls in CLASSES:
        path = os.path.join(BASE_DIR, cls)
        n = len([f for f in os.listdir(path) if f.endswith('.npy')]) if os.path.exists(path) else 0
        bar = "#" * min(n // 5, 30)
        print(f"  {cls:10s}: {n:4d}  {bar}")
    print("========================================")

def main():
    """Main data collection loop with single and burst capture modes."""
    setup()
    print("Connecting to ESP32...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        sock.connect((ESP32_IP, ESP32_PORT))
        print("Connected!")
        
        # Take test shot to verify image quality
        print("Taking test shot...")
        img = capture_fresh(sock)
        if img:
            bpp = struct.unpack_from('<H', img, 28)[0]
            w = struct.unpack_from('<i', img, 18)[0]
            h = struct.unpack_from('<i', img, 22)[0]
            print(f"  BMP: {w}x{h}, {bpp}-bit")
            pixels = process_raw_image(img)
            if pixels is not None:
                print(f"  Pixel range: min={pixels.min():.3f} max={pixels.max():.3f} mean={pixels.mean():.3f} std={pixels.std():.3f}")

        while True:
            show_status()
            print("\nCommands: r/p/s = capture, q = quit")
            choice = input("Choice: ").strip().lower()

            if choice == 'q':
                sock.send(b"quit")
                break
            elif choice in ['r', 'p', 's']:
                cls = {"r": "rock", "p": "paper", "s": "scissors"}[choice]
                print(f"\nCapturing [{cls.upper()}]")
                print("  ENTER = capture one, number = burst, done = go back\n")

                while True:
                    cmd = input(f"  [{cls}] > ").strip().lower()
                    if cmd == 'done':
                        break
                    elif cmd == '':
                        # Single capture
                        img_data = capture_fresh(sock)
                        if img_data:
                            pixels = process_raw_image(img_data)
                            if pixels is not None:
                                num = counters[cls]
                                counters[cls] += 1
                                filepath = os.path.join(BASE_DIR, cls, f"{cls}_{num:04d}.npy")
                                np.save(filepath, pixels)
                                total = len([f for f in os.listdir(os.path.join(BASE_DIR, cls)) if f.endswith('.npy')])
                                print(f"    Saved! ({total} total {cls})")
                    elif cmd.isdigit():
                        # Burst capture — capture N images rapidly
                        num = int(cmd)
                        print(f"    Burst: {num} images. Hold steady!")
                        time.sleep(0.5)
                        for i in range(num):
                            img_data = capture_fresh(sock)
                            if img_data:
                                pixels = process_raw_image(img_data)
                                if pixels is not None:
                                    n = counters[cls]
                                    counters[cls] += 1
                                    filepath = os.path.join(BASE_DIR, cls, f"{cls}_{n:04d}.npy")
                                    np.save(filepath, pixels)
                                    total = len([f for f in os.listdir(os.path.join(BASE_DIR, cls)) if f.endswith('.npy')])
                                    print(f"    [{i+1}/{num}] Saved ({total} total)")
                            time.sleep(0.1)
                        print(f"    Done!")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()
        show_status()

if __name__ == "__main__":
    main()
