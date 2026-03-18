# classify.py — runs on ESP32
# Captures an image and classifies it using the CNN

from camera import Camera, GrabMode, PixelFormat, FrameSize, GainCeiling
from image_preprocessing import resize_96x96_to_32x32_and_threshold
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
}

# Initialize camera
cam = Camera(**CAMERA_PARAMETERS)
cam.init()
cam.set_bmp_out(True)
print("Camera ready")

# Load CNN model
model = TinyCNN()

# Brightness and contrast settings (must match training data!)
BRIGHTNESS_BOOST = 30
CONTRAST_FACTOR = 1.5

def extract_pixels(bmp_data):
    """Extract pixel values from 32x32 BMP, apply brightness/contrast, normalize"""
    data_offset = struct.unpack_from('<I', bmp_data, 10)[0]
    width = 32
    height = 32
    row_size = ((width * 8 + 31) // 32) * 4

    pixels = []
    for y in range(height - 1, -1, -1):  # BMP is bottom-up
        for x in range(width):
            offset = data_offset + y * row_size + x
            if offset < len(bmp_data):
                val = bmp_data[offset]
                # Apply same brightness/contrast as training
                val = val + BRIGHTNESS_BOOST
                val = int((val - 128) * CONTRAST_FACTOR + 128)
                val = max(0, min(255, val))
                pixels.append(val / 255.0)
            else:
                pixels.append(0.0)
    return pixels

def classify():
    """Capture and classify one image"""
    # Discard stale frames from buffer
    for _ in range(3):
        cam.capture()
        time.sleep(0.1)

    # Now capture the real frame
    img = cam.capture()

    # Resize to 32x32
    resized = resize_96x96_to_32x32_and_threshold(img, -1)

    # Extract pixels
    pixels = extract_pixels(resized)

    # Classify
    label, confidence, probs = model.predict(pixels)

    return label, confidence, probs

# Main loop
print("\nReady! Press Enter to classify.")
while True:
    input("Press Enter...")
    label, conf, probs = classify()
    print(f"\n  Prediction: {label.upper()}")
    print(f"  Confidence: {conf * 100:.1f}%")
    print(f"  Rock: {probs[0]*100:.1f}%  Paper: {probs[1]*100:.1f}%  Scissors: {probs[2]*100:.1f}%")
    print()
