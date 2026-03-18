# collect_v2.py — runs on your LAPTOP
# Collects training data using EXACT same processing as classify_laptop.py
# This guarantees training images match inference images perfectly
#
# Usage:
#   1. Run socket_server.py on ESP32
#   2. python3 collect_v2.py

import socket
import struct
import os
import time
import numpy as np

# --- Settings ---
ESP32_IP = "172.20.10.4"  # UPDATE THIS
ESP32_PORT = 8080

BASE_DIR = "training_data_v2"
CLASSES = ["rock", "paper", "scissors"]
BRIGHTNESS_BOOST = 30
CONTRAST_FACTOR = 1.5

counters = {"rock": 0, "paper": 0, "scissors": 0}

def setup():
    for cls in CLASSES:
        path = os.path.join(BASE_DIR, cls)
        os.makedirs(path, exist_ok=True)
        files = [f for f in os.listdir(path) if f.endswith('.npy')]
        if files:
            nums = []
            for f in files:
                try:
                    nums.append(int(f.replace(cls + "_", "").replace(".npy", "")))
                except:
                    pass
            counters[cls] = max(nums) + 1 if nums else 0

def receive_image(sock):
    size_data = sock.recv(4)
    if len(size_data) < 4:
        return None
    img_size = int.from_bytes(size_data, 'big')
    img_data = b""
    while len(img_data) < img_size:
        chunk = sock.recv(min(4096, img_size - len(img_data)))
        if not chunk:
            break
        img_data += chunk
    return img_data

def process_image(bmp_data):
    """Process image EXACTLY like classify_laptop.py does"""
    if bmp_data[0:2] != b'BM':
        return None

    data_offset = struct.unpack_from('<I', bmp_data, 10)[0]
    width = struct.unpack_from('<i', bmp_data, 18)[0]
    height = struct.unpack_from('<i', bmp_data, 22)[0]
    bits_per_pixel = struct.unpack_from('<H', bmp_data, 28)[0]

    abs_height = abs(height)
    row_size = ((width * bits_per_pixel + 31) // 32) * 4

    pixels = []
    for y in range(abs_height):
        row = []
        for x in range(width):
            offset = data_offset + y * row_size + x * (bits_per_pixel // 8)
            if offset < len(bmp_data):
                pixel = bmp_data[offset] & 0xFF
            else:
                pixel = 0
            row.append(pixel)
        pixels.append(row)

    # Resize to 32x32
    resized = np.zeros((32, 32), dtype=np.float32)
    for new_y in range(32):
        old_y = (new_y * abs_height) // 32
        for new_x in range(32):
            old_x = (new_x * width) // 32
            if old_y < len(pixels) and old_x < len(pixels[old_y]):
                val = pixels[old_y][old_x]
            else:
                val = 0
            val = val + BRIGHTNESS_BOOST
            val = int((val - 128) * CONTRAST_FACTOR + 128)
            val = max(0, min(255, val))
            resized[new_y][new_x] = val / 255.0

    return resized

def capture_fresh(sock):
    """Discard stale frames and get a fresh one"""
    for _ in range(2):
        sock.send(b"capture")
        receive_image(sock)
        time.sleep(0.1)
    sock.send(b"capture")
    return receive_image(sock)

def show_status():
    print("\n========================================")
    for cls in CLASSES:
        path = os.path.join(BASE_DIR, cls)
        n = len([f for f in os.listdir(path) if f.endswith('.npy')]) if os.path.exists(path) else 0
        bar = "#" * min(n // 5, 30)
        print(f"  {cls:10s}: {n:4d}  {bar}")
    print("========================================")

def main():
    setup()
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
                        img_data = capture_fresh(sock)
                        if img_data:
                            pixels = process_image(img_data)
                            if pixels is not None:
                                num = counters[cls]
                                counters[cls] += 1
                                filepath = os.path.join(BASE_DIR, cls, f"{cls}_{num:04d}.npy")
                                np.save(filepath, pixels)
                                total = len([f for f in os.listdir(os.path.join(BASE_DIR, cls)) if f.endswith('.npy')])
                                print(f"    Saved! ({total} total {cls})")
                    elif cmd.isdigit():
                        num = int(cmd)
                        print(f"    Burst: {num} images. Hold steady!")
                        time.sleep(0.5)
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
                            time.sleep(0.1)
                        print(f"    Done!")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()
        show_status()

if __name__ == "__main__":
    main()
