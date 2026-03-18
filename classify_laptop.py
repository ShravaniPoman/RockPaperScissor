# classify_laptop.py — runs on your LAPTOP
# Receives images from ESP32 and classifies them
# Uses EXACT same processing as collect_data.py

import socket
import struct
import numpy as np
import time

# --- Settings ---
ESP32_IP = "172.20.10.4"  # UPDATE THIS
ESP32_PORT = 8080

CLASSES = ["rock", "paper", "scissors"]
IMG_SIZE = 32
BRIGHTNESS_BOOST = 30
CONTRAST_FACTOR = 1.5

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

def resize_bmp_to_32x32(bmp_data, brightness_boost=30, contrast_factor=1.5):
    """
    EXACT same function as collect_data.py used during training
    """
    if bmp_data[0:2] != b'BM':
        return None

    data_offset = struct.unpack_from('<I', bmp_data, 10)[0]
    width = struct.unpack_from('<i', bmp_data, 18)[0]
    height = struct.unpack_from('<i', bmp_data, 22)[0]
    bits_per_pixel = struct.unpack_from('<H', bmp_data, 28)[0]

    NEW_W = 32
    NEW_H = 32

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

    resized = []
    for new_y in range(NEW_H):
        row = []
        old_y = (new_y * abs_height) // NEW_H
        for new_x in range(NEW_W):
            old_x = (new_x * width) // NEW_W
            if old_y < len(pixels) and old_x < len(pixels[old_y]):
                val = pixels[old_y][old_x]
            else:
                val = 0
            val = val + brightness_boost
            val = int((val - 128) * contrast_factor + 128)
            val = max(0, min(255, val))
            row.append(val)
        resized.append(row)

    # Build BMP file (for saving debug images)
    HEADER_SIZE = 14
    DIB_SIZE = 40
    PALETTE_SIZE = 256 * 4
    ROW_PADDING = (NEW_W % 4)
    PIXEL_DATA_SIZE = (NEW_W + ROW_PADDING) * NEW_H
    FILE_SIZE = HEADER_SIZE + DIB_SIZE + PALETTE_SIZE + PIXEL_DATA_SIZE

    bmp = bytearray(FILE_SIZE)
    bmp[0:2] = b'BM'
    struct.pack_into('<I', bmp, 2, FILE_SIZE)
    struct.pack_into('<I', bmp, 10, HEADER_SIZE + DIB_SIZE + PALETTE_SIZE)
    struct.pack_into('<I', bmp, 14, DIB_SIZE)
    struct.pack_into('<i', bmp, 18, NEW_W)
    struct.pack_into('<i', bmp, 22, NEW_H)
    struct.pack_into('<H', bmp, 26, 1)
    struct.pack_into('<H', bmp, 28, 8)
    struct.pack_into('<I', bmp, 34, PIXEL_DATA_SIZE)
    for i in range(256):
        off = HEADER_SIZE + DIB_SIZE + i * 4
        bmp[off] = i
        bmp[off+1] = i
        bmp[off+2] = i
        bmp[off+3] = 0

    pixel_offset = HEADER_SIZE + DIB_SIZE + PALETTE_SIZE
    for y in range(NEW_H):
        for x in range(NEW_W):
            bmp[pixel_offset] = resized[NEW_H - 1 - y][x]
            pixel_offset += 1
        pixel_offset += ROW_PADDING

    return resized, bytes(bmp)

def main():
    print("Loading model...")
    import tensorflow as tf
    model = tf.keras.models.load_model('rps_tiny_model.keras')
    print("Model loaded!")

    print(f"\nConnecting to ESP32 at {ESP32_IP}:{ESP32_PORT}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        sock.connect((ESP32_IP, ESP32_PORT))
        print("Connected!")
        print("\nReady! Press Enter to capture and classify.")
        print("Type 'q' to quit.\n")

        frame_count = 0
        while True:
            cmd = input("Press Enter (or 'q' to quit)... ").strip()
            if cmd == 'q':
                sock.send(b"quit")
                break

            # Discard stale frames
            for _ in range(2):
                sock.send(b"capture")
                receive_image(sock)
                time.sleep(0.1)

            # Capture fresh frame
            sock.send(b"capture")
            img_data = receive_image(sock)

            if img_data:
                # Process EXACTLY like collect_data.py did
                result = resize_bmp_to_32x32(img_data, BRIGHTNESS_BOOST, CONTRAST_FACTOR)
                if result is None:
                    print("  Failed to process image!")
                    continue

                resized_pixels, bmp_bytes = result

                # Save debug image
                debug_file = f"debug_frame_{frame_count}.bmp"
                with open(debug_file, "wb") as f:
                    f.write(bmp_bytes)

                # Convert to numpy for model - same as training
                pixels_flat = []
                for row in resized_pixels:
                    for val in row:
                        pixels_flat.append(val / 255.0)

                img_input = np.array(pixels_flat, dtype=np.float32).reshape(1, IMG_SIZE, IMG_SIZE, 1)

                # Predict
                pred = model.predict(img_input, verbose=0)
                probs = pred[0]
                best_idx = np.argmax(probs)
                label = CLASSES[best_idx]
                conf = probs[best_idx]

                print(f"\n  Prediction: {label.upper()}")
                print(f"  Confidence: {conf * 100:.1f}%")
                print(f"  Rock: {probs[0]*100:.1f}%  Paper: {probs[1]*100:.1f}%  Scissors: {probs[2]*100:.1f}%")
                print(f"  (Saved debug image: {debug_file})")
                print()
                frame_count += 1
            else:
                print("  Failed to capture!")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()
        print("Done!")

if __name__ == "__main__":
    main()
