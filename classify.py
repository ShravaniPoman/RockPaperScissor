# classify.py — Runs on the ESP32 (MicroPython)
# Purpose: On-device gesture classification. Captures an image from the camera,
#          resizes it to 32x32, and runs CNN inference entirely on the ESP32
#          using the TinyCNN engine from cnn_model.py (no laptop needed).
#
# Requires on ESP32: cnn_model.py, model_esp32.bin, image_preprocessing.py
#
# Note: Pure Python inference is slow on the ESP32 (~5-10 seconds per classification).
#       For real-time use, classify_laptop.py is faster since TensorFlow runs on the laptop.
#
# Author: Shravani Poman (with assistance from Claude AI)
# Course: AI for Engineers, Spring 2026

from camera import Camera, GrabMode, PixelFormat, FrameSize, GainCeiling
from image_preprocessing import resize_96x96_to_32x32_and_threshold  # SEEED Studio
from cnn_model import TinyCNN  # Our custom MicroPython CNN inference engine
import struct
import time

# --- Camera Pin Configuration for XIAO ESP32-S3 Sense ---
CAMERA_PARAMETERS = {
    "data_pins": [15, 17, 18, 16, 14, 12, 11, 48],  # 8-bit parallel data bus
    "vsync_pin": 38,       # Vertical sync
    "href_pin": 47,        # Horizontal reference
    "sda_pin": 40,         # I2C data
    "scl_pin": 39,         # I2C clock
    "pclk_pin": 13,        # Pixel clock
    "xclk_pin": 10,        # Master clock
    "xclk_freq": 20000000, # 20MHz
    "powerdown_pin": -1,   # Not used
    "reset_pin": -1,       # Not used
}

# Initialize camera in BMP grayscale mode
cam = Camera(**CAMERA_PARAMETERS)
cam.init()
cam.set_bmp_out(True)  # 8-bit grayscale BMP output
print("Camera ready")

# Load the CNN model — reads weights from model_esp32.bin into memory
model = TinyCNN()

# --- Image Enhancement Settings ---
# These MUST match the values used during training (in collect_v2.py and classify_laptop.py)
# Without matching these, the model sees different-looking images than it was trained on
BRIGHTNESS_BOOST = 30   # Added to each pixel to compensate for dark camera images
CONTRAST_FACTOR = 1.5   # Multiplied around midpoint 128 to increase hand visibility

def extract_pixels(bmp_data):
    """Extract pixel values from a 32x32 BMP and prepare them for CNN input.
    
    Reads the BMP pixel data, applies the same brightness/contrast enhancement
    used during training, and normalizes to 0.0-1.0 float values.
    
    BMP files store rows bottom-up, so we iterate from the last row to the first
    to get top-down pixel order that matches the training data format.
    
    Args:
        bmp_data: Raw bytes of a 32x32 8-bit grayscale BMP file
    Returns:
        List of 1024 float values (32x32 pixels, 0.0-1.0 range)
    """
    # Read pixel data offset from BMP header (bytes 10-13, little-endian)
    data_offset = struct.unpack_from('<I', bmp_data, 10)[0]
    width = 32
    height = 32
    # BMP rows are padded to 4-byte boundaries
    row_size = ((width * 8 + 31) // 32) * 4

    pixels = []
    # Read pixels bottom-up (BMP format) to get top-down order
    for y in range(height - 1, -1, -1):
        for x in range(width):
            offset = data_offset + y * row_size + x
            if offset < len(bmp_data):
                val = bmp_data[offset]
                # Apply brightness boost (same as training: +30)
                val = val + BRIGHTNESS_BOOST
                # Apply contrast enhancement (same as training: x1.5 around midpoint 128)
                val = int((val - 128) * CONTRAST_FACTOR + 128)
                # Clamp to valid pixel range [0, 255]
                val = max(0, min(255, val))
                # Normalize to 0.0-1.0 for neural network input
                pixels.append(val / 255.0)
            else:
                pixels.append(0.0)
    return pixels

def classify():
    """Capture a fresh image from the camera and classify it.
    
    Process:
    1. Discard 3 stale frames from camera buffer (prevents using old cached images)
    2. Capture a fresh frame
    3. Resize from camera resolution to 32x32 using SEEED's preprocessing
    4. Extract and enhance pixels (matching training pipeline)
    5. Run CNN forward pass to get class prediction
    
    Returns:
        (label, confidence, probabilities) — e.g. ("rock", 0.98, [0.98, 0.01, 0.01])
    """
    # Discard stale frames — the ESP32 camera buffers old frames
    # Without this, we'd classify an old image, not what's currently in view
    for _ in range(3):
        cam.capture()
        time.sleep(0.1)

    # Capture the actual frame we want to classify
    img = cam.capture()

    # Resize to 32x32 using SEEED's preprocessing function
    # threshold=-1 means keep full grayscale (no binary thresholding)
    resized = resize_96x96_to_32x32_and_threshold(img, -1)

    # Extract pixels with brightness/contrast enhancement and normalization
    pixels = extract_pixels(resized)

    # Run CNN inference — this is the slow part on ESP32 (~5-10 seconds)
    # The predict() method runs: Conv2D → Pool → Conv2D → Pool → Conv2D → Pool → Dense → Dense → Softmax
    label, confidence, probs = model.predict(pixels)

    return label, confidence, probs

# --- Main Classification Loop ---
# Waits for user to press Enter, then captures and classifies one image
print("\nReady! Press Enter to classify.")
print("(Note: Each classification takes ~5-10 seconds on ESP32)")
while True:
    input("Press Enter...")
    label, conf, probs = classify()
    # Display the prediction with confidence percentages for all three classes
    print(f"\n  Prediction: {label.upper()}")
    print(f"  Confidence: {conf * 100:.1f}%")
    print(f"  Rock: {probs[0]*100:.1f}%  Paper: {probs[1]*100:.1f}%  Scissors: {probs[2]*100:.1f}%")
    print()
