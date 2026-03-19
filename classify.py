# classify.py — Runs on the ESP32 (MicroPython)
# On-device classification with grayscale camera output
# IMPORTANT: Uses & 0xFF to convert signed bytes to unsigned (0-255)
# MicroPython returns signed bytes from BMP data, laptop Python returns unsigned
#
# Author: Shravani Poman (with assistance from Claude AI)

from camera import Camera, GrabMode, PixelFormat, FrameSize, GainCeiling
from cnn_model import TinyCNN
import struct
import time

CAMERA_PARAMETERS = {
    "data_pins": [15, 17, 18, 16, 14, 12, 11, 48],
    "vsync_pin": 38,
    "href_pin": 47,
    "sda_pin": 40,
    "scl_pin": 39,
    "pclk_pin": 13,
    "xclk_pin": 10,
    "xclk_freq": 20000000,
    "powerdown_pin": -1,
    "reset_pin": -1,
    "pixel_format": PixelFormat.GRAYSCALE,
}

cam = Camera(**CAMERA_PARAMETERS)
cam.init()
cam.set_bmp_out(True)
print("Camera ready (grayscale mode)")

model = TinyCNN()

def resize_and_extract(bmp_data):
    """Resize 8-bit grayscale BMP to 32x32 and normalize.
    Uses & 0xFF to ensure unsigned byte values (0-255).
    MicroPython returns signed bytes (-128 to 127) from BMP data."""
    
    data_offset = struct.unpack_from('<I', bmp_data, 10)[0]
    width = struct.unpack_from('<i', bmp_data, 18)[0]
    height = struct.unpack_from('<i', bmp_data, 22)[0]
    
    abs_height = abs(height)
    top_down = height < 0
    row_size = ((width * 8 + 31) // 32) * 4
    
    # Read raw pixels — use & 0xFF to convert signed to unsigned!
    raw = []
    for y in range(abs_height):
        for x in range(width):
            offset = data_offset + y * row_size + x
            if offset < len(bmp_data):
                raw.append(bmp_data[offset] & 0xFF)  # CRITICAL: unsigned conversion
            else:
                raw.append(0)
    
    # Resize to 32x32 nearest-neighbor
    pixels = []
    for new_y in range(32):
        if top_down:
            old_y = (new_y * abs_height) // 32
        else:
            old_y = abs_height - 1 - (new_y * abs_height) // 32
        
        for new_x in range(32):
            old_x = (new_x * width) // 32
            idx = old_y * width + old_x
            if idx < len(raw):
                pixels.append(raw[idx] / 255.0)
            else:
                pixels.append(0.0)
    
    return pixels

def classify():
    """Capture and classify one image."""
    for _ in range(3):
        cam.capture()
        time.sleep(0.1)
    
    img = cam.capture()
    pixels = resize_and_extract(img)
    label, confidence, probs = model.predict(pixels)
    return label, confidence, probs

print("\nReady! Press Enter to classify.")
print("Classification runs entirely on ESP32.")
while True:
    input("Press Enter...")
    label, conf, probs = classify()
    print(f"\n  Prediction: {label.upper()}")
    print(f"  Confidence: {conf * 100:.1f}%")
    print(f"  Rock: {probs[0]*100:.1f}%  Paper: {probs[1]*100:.1f}%  Scissors: {probs[2]*100:.1f}%")
    print()
