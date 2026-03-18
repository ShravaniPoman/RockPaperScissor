# classify_laptop.py — Runs on your LAPTOP
# Purpose: Real-time rock/paper/scissors classification. Receives images from
#          the ESP32 camera over WiFi and classifies them using the trained CNN.
#
# CRITICAL: Uses the EXACT same image processing pipeline as collect_v2.py
# (resize, brightness, contrast, normalization) to ensure training/inference match.
#
# Also saves debug images (debug_frame_N.bmp) for verifying what the model sees.
#
# Usage:
#   1. Run socket_server.py on ESP32 (with WiFi connected)
#   2. python3 classify_laptop.py
#   3. Press Enter to capture and classify, 'q' to quit
#
# Author: Shravani Poman (with assistance from Claude AI)


import socket
import struct
import numpy as np
import time

# --- Connection Settings ---
ESP32_IP = "172.20.10.4"  # ESP32's IP address — update if it changes
ESP32_PORT = 8080          # Must match socket_server.py PORT

# --- Classification Settings ---
CLASSES = ["rock", "paper", "scissors"]  # Class labels matching training order
IMG_SIZE = 32  # CNN input size (32x32 pixels)

# --- Image Enhancement Settings ---
# These MUST match collect_v2.py values exactly for pipeline consistency
BRIGHTNESS_BOOST = 30   # Added to each raw pixel (compensates for dark camera)
CONTRAST_FACTOR = 1.5   # Applied around midpoint 128 (enhances hand visibility)

def receive_image(sock):
    """Receive a BMP image from the ESP32 socket server.
    Protocol: 4 bytes (big-endian image size) + raw BMP data.
    Reads in chunks until the full image is received."""
    size_data = sock.recv(4)  # Read the 4-byte size header
    if len(size_data) < 4:
        return None
    img_size = int.from_bytes(size_data, 'big')  # Decode size
    # Accumulate image data in chunks (network may split large transfers)
    img_data = b""
    while len(img_data) < img_size:
        chunk = sock.recv(min(4096, img_size - len(img_data)))
        if not chunk:
            break
        img_data += chunk
    return img_data

def resize_bmp_to_32x32(bmp_data, brightness_boost=30, contrast_factor=1.5):
    """Resize a raw BMP image to 32x32 with brightness/contrast enhancement.
    
    This function implements the EXACT same processing as collect_v2.py's
    process_image() function. Both must produce identical output for the
    same input, or the model will fail on live images.
    
    Steps:
    1. Parse BMP header (data offset, width, height, bits per pixel)
    2. Read raw pixel values into 2D array
    3. Resize to 32x32 using nearest-neighbor interpolation
    4. Apply brightness boost and contrast enhancement
    5. Build a new 32x32 BMP file for debug saving
    
    Args:
        bmp_data: Raw BMP file bytes from camera
        brightness_boost: Value added to each pixel (default 30)
        contrast_factor: Multiplier around midpoint 128 (default 1.5)
    
    Returns: (resized_pixels_2d_list, bmp_bytes) or None if invalid
    """
    # Validate BMP file signature
    if bmp_data[0:2] != b'BM':
        return None

    # Parse BMP header fields (little-endian byte order)
    data_offset = struct.unpack_from('<I', bmp_data, 10)[0]   # Where pixel data starts
    width = struct.unpack_from('<i', bmp_data, 18)[0]          # Image width
    height = struct.unpack_from('<i', bmp_data, 22)[0]         # Image height (negative = top-down)
    bits_per_pixel = struct.unpack_from('<H', bmp_data, 28)[0] # 8 for grayscale

    NEW_W = 32  # Target width
    NEW_H = 32  # Target height

    abs_height = abs(height)
    # BMP rows are padded to 4-byte boundaries
    row_size = ((width * bits_per_pixel + 31) // 32) * 4

    # Read all pixel values from BMP into a 2D list
    pixels = []
    for y in range(abs_height):
        row = []
        for x in range(width):
            offset = data_offset + y * row_size + x * (bits_per_pixel // 8)
            if offset < len(bmp_data):
                pixel = bmp_data[offset] & 0xFF  # Unsigned 8-bit value
            else:
                pixel = 0
            row.append(pixel)
        pixels.append(row)

    # Resize to 32x32 using nearest-neighbor interpolation
    # For each output pixel, find the closest input pixel
    resized = []
    for new_y in range(NEW_H):
        row = []
        old_y = (new_y * abs_height) // NEW_H  # Map output Y to input Y
        for new_x in range(NEW_W):
            old_x = (new_x * width) // NEW_W    # Map output X to input X
            if old_y < len(pixels) and old_x < len(pixels[old_y]):
                val = pixels[old_y][old_x]
            else:
                val = 0
            # Apply brightness boost (add fixed value to brighten dark images)
            val = val + brightness_boost
            # Apply contrast enhancement: stretch values away from midpoint
            # Formula: new_val = (old_val - 128) * factor + 128
            val = int((val - 128) * contrast_factor + 128)
            # Clamp to valid pixel range [0, 255]
            val = max(0, min(255, val))
            row.append(val)
        resized.append(row)

    # --- Build a 32x32 BMP file for debug visualization ---
    # This allows us to save what the model actually "sees" for debugging
    HEADER_SIZE = 14       # BMP file header
    DIB_SIZE = 40          # DIB (image info) header
    PALETTE_SIZE = 256 * 4 # 256-color grayscale palette (R,G,B,reserved for each)
    ROW_PADDING = (NEW_W % 4)  # BMP rows must be 4-byte aligned
    PIXEL_DATA_SIZE = (NEW_W + ROW_PADDING) * NEW_H
    FILE_SIZE = HEADER_SIZE + DIB_SIZE + PALETTE_SIZE + PIXEL_DATA_SIZE

    bmp = bytearray(FILE_SIZE)
    # BMP file header
    bmp[0:2] = b'BM'  # Magic signature
    struct.pack_into('<I', bmp, 2, FILE_SIZE)       # Total file size
    struct.pack_into('<I', bmp, 10, HEADER_SIZE + DIB_SIZE + PALETTE_SIZE)  # Pixel data offset
    # DIB header
    struct.pack_into('<I', bmp, 14, DIB_SIZE)       # DIB header size
    struct.pack_into('<i', bmp, 18, NEW_W)          # Width
    struct.pack_into('<i', bmp, 22, NEW_H)          # Height (positive = bottom-up)
    struct.pack_into('<H', bmp, 26, 1)              # Color planes (always 1)
    struct.pack_into('<H', bmp, 28, 8)              # Bits per pixel (8 = grayscale)
    struct.pack_into('<I', bmp, 34, PIXEL_DATA_SIZE) # Raw pixel data size
    
    # Write grayscale palette (256 entries: gray value → R=G=B=gray)
    for i in range(256):
        off = HEADER_SIZE + DIB_SIZE + i * 4
        bmp[off] = i      # Blue
        bmp[off+1] = i    # Green
        bmp[off+2] = i    # Red
        bmp[off+3] = 0    # Reserved

    # Write pixel data (BMP stores rows bottom-up)
    pixel_offset = HEADER_SIZE + DIB_SIZE + PALETTE_SIZE
    for y in range(NEW_H):
        for x in range(NEW_W):
            bmp[pixel_offset] = resized[NEW_H - 1 - y][x]  # Flip vertically
            pixel_offset += 1
        pixel_offset += ROW_PADDING  # Pad each row to 4-byte boundary

    return resized, bytes(bmp)

def main():
    """Main classification loop: connect to ESP32, capture images, classify."""
    # Load the trained TensorFlow/Keras model
    print("Loading model...")
    import tensorflow as tf
    model = tf.keras.models.load_model('rps_tiny_model.keras')
    print("Model loaded!")

    # Connect to ESP32's socket server
    print(f"\nConnecting to ESP32 at {ESP32_IP}:{ESP32_PORT}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        sock.connect((ESP32_IP, ESP32_PORT))
        print("Connected!")
        print("\nReady! Press Enter to capture and classify.")
        print("Type 'q' to quit.\n")

        frame_count = 0  # Counter for debug image filenames
        while True:
            cmd = input("Press Enter (or 'q' to quit)... ").strip()
            if cmd == 'q':
                sock.send(b"quit")  # Tell ESP32 server we're disconnecting
                break

            # Discard 2 stale frames from camera buffer
            # The ESP32 camera buffers frames, so first captures may be old
            for _ in range(2):
                sock.send(b"capture")
                receive_image(sock)  # Receive and throw away
                time.sleep(0.1)

            # Capture a fresh frame
            sock.send(b"capture")
            img_data = receive_image(sock)

            if img_data:
                # Process image through SAME pipeline as training (collect_v2.py)
                result = resize_bmp_to_32x32(img_data, BRIGHTNESS_BOOST, CONTRAST_FACTOR)
                if result is None:
                    print("  Failed to process image!")
                    continue

                resized_pixels, bmp_bytes = result

                # Save debug BMP image so we can visually inspect what model sees
                debug_file = f"debug_frame_{frame_count}.bmp"
                with open(debug_file, "wb") as f:
                    f.write(bmp_bytes)

                # Convert 2D pixel list to normalized numpy array for the model
                # Flatten row by row, divide by 255 to get 0.0-1.0 range
                pixels_flat = []
                for row in resized_pixels:
                    for val in row:
                        pixels_flat.append(val / 255.0)

                # Reshape to model's expected input: (batch=1, height=32, width=32, channels=1)
                img_input = np.array(pixels_flat, dtype=np.float32).reshape(1, IMG_SIZE, IMG_SIZE, 1)

                # Run CNN inference — returns probability array [rock, paper, scissors]
                pred = model.predict(img_input, verbose=0)
                probs = pred[0]  # First (only) image in batch
                best_idx = np.argmax(probs)  # Index of highest probability
                label = CLASSES[best_idx]     # Map index to class name
                conf = probs[best_idx]        # Confidence = highest probability

                # Display results
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
