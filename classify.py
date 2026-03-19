# classify.py — Runs on the ESP32 (MicroPython)
#
# Purpose: On-device rock/paper/scissors classification.
#          Captures a grayscale image from the camera, resizes to 32x32,
#          runs CNN inference using cnn_model.py, and prints the result
#          to the serial monitor (Thonny). No laptop processing needed.
#
# Requirements on ESP32: cnn_model.py, model_esp32.bin
#
# Note: Each classification takes ~5-10 seconds because the CNN forward pass
#       runs in pure Python on the microcontroller.
#
# Author: Shravani Poman (with assistance from Claude AI)
# Course: AI for Engineers, Spring 2026

from camera import Camera, GrabMode, PixelFormat, FrameSize, GainCeiling
from cnn_model import TinyCNN  # Custom MicroPython CNN inference engine
import struct
import time

# --- Camera Pin Configuration for XIAO ESP32-S3 Sense ---
# These pin assignments match the hardware wiring between the ESP32-S3
# processor and the OV3660 camera module on the XIAO Sense board
CAMERA_PARAMETERS = {
    "data_pins": [15, 17, 18, 16, 14, 12, 11, 48],  # 8-bit parallel data bus
    "vsync_pin": 38,       # Vertical sync signal
    "href_pin": 47,        # Horizontal reference signal
    "sda_pin": 40,         # I2C data for camera register configuration
    "scl_pin": 39,         # I2C clock for camera register configuration
    "pclk_pin": 13,        # Pixel clock from camera
    "xclk_pin": 10,        # Master clock output to camera
    "xclk_freq": 20000000, # 20MHz master clock frequency
    "powerdown_pin": -1,   # Not connected on this board
    "reset_pin": -1,       # Not connected on this board
    "pixel_format": PixelFormat.GRAYSCALE,  # Force 8-bit grayscale output (not 24-bit color)
}

# Initialize camera in grayscale BMP mode
cam = Camera(**CAMERA_PARAMETERS)
cam.init()  # Power on camera and configure via I2C
cam.set_bmp_out(True)  # Output as BMP format with header
print("Camera ready (grayscale mode)")

# Load the CNN model — reads weights from model_esp32.bin into memory
# Model: 3x Conv2D(12 filters) + Dense(24) + Dense(3), ~7,443 parameters
model = TinyCNN()

def resize_and_extract(bmp_data):
    """Resize a 128x128 grayscale BMP to 32x32 and normalize for CNN input.
    
    CRITICAL DETAILS:
    - Uses & 0xFF to convert MicroPython signed bytes to unsigned (0-255)
      Without this, pixel values >127 become negative, causing wrong predictions
    - Handles negative height (top-down BMP storage from OV3660 camera)
    - Must match collect_esp.py's process_raw_image() exactly for correct results
    
    Args:
        bmp_data: Raw BMP bytes from cam.capture()
    Returns:
        Flat list of 1024 float values (32x32 pixels, 0.0-1.0 range)
    """
    # Parse BMP header to get image dimensions and pixel data location
    data_offset = struct.unpack_from('<I', bmp_data, 10)[0]  # Where pixel data starts (1078 for 8-bit)
    width = struct.unpack_from('<i', bmp_data, 18)[0]         # Image width (128)
    height = struct.unpack_from('<i', bmp_data, 22)[0]        # Height (-128 = top-down storage)
    
    abs_height = abs(height)
    top_down = height < 0  # OV3660 outputs negative height (top-down BMP)
    # BMP rows padded to 4-byte boundary (128 pixels * 1 byte = 128, already aligned)
    row_size = ((width * 8 + 31) // 32) * 4
    
    # Read all raw pixels into a flat list
    # CRITICAL: Use & 0xFF to convert signed bytes to unsigned!
    # MicroPython returns signed (-128 to 127), we need unsigned (0 to 255)
    raw = []
    for y in range(abs_height):
        for x in range(width):
            offset = data_offset + y * row_size + x
            if offset < len(bmp_data):
                raw.append(bmp_data[offset] & 0xFF)  # Unsigned conversion
            else:
                raw.append(0)
    
    # Resize from 128x128 to 32x32 using nearest-neighbor interpolation
    # Each output pixel maps to the closest input pixel
    pixels = []
    for new_y in range(32):
        if top_down:
            # Top-down BMP: row 0 in file = top of image (correct order)
            old_y = (new_y * abs_height) // 32
        else:
            # Bottom-up BMP: row 0 in file = bottom of image (flip needed)
            old_y = abs_height - 1 - (new_y * abs_height) // 32
        
        for new_x in range(32):
            old_x = (new_x * width) // 32  # Map output X to input X
            idx = old_y * width + old_x     # 1D index into flat pixel array
            if idx < len(raw):
                pixels.append(raw[idx] / 255.0)  # Normalize to 0.0-1.0 for CNN
            else:
                pixels.append(0.0)
    
    return pixels

def classify():
    """Capture a fresh image and run CNN classification.
    
    Steps:
    1. Discard 3 stale frames (camera buffers old images)
    2. Capture fresh frame
    3. Resize to 32x32 and normalize pixels
    4. Run CNN forward pass (Conv2D -> Pool -> Dense -> Softmax)
    
    Returns: (label, confidence, probabilities)
    """
    # Discard stale frames — the camera buffer may contain old images
    for _ in range(3):
        cam.capture()
        time.sleep(0.1)
    
    # Capture the actual frame to classify
    img = cam.capture()
    
    # Resize and extract normalized pixel values
    pixels = resize_and_extract(img)
    
    # Run CNN inference — returns class name, confidence, and all probabilities
    label, confidence, probs = model.predict(pixels)
    return label, confidence, probs

# --- Main Classification Loop ---
# Waits for user input, then captures and classifies one image
print("\nReady! Press Enter to classify.")
print("Classification runs entirely on ESP32.")
while True:
    input("Press Enter...")
    label, conf, probs = classify()
    # Display prediction with confidence for all three classes
    print(f"\n  Prediction: {label.upper()}")
    print(f"  Confidence: {conf * 100:.1f}%")
    print(f"  Rock: {probs[0]*100:.1f}%  Paper: {probs[1]*100:.1f}%  Scissors: {probs[2]*100:.1f}%")
    print()
