# socket_server.py — Runs on the ESP32 (MicroPython)
#
# Purpose: WiFi camera server that captures 8-bit grayscale images from the
#          OV3660 camera and sends them to a laptop client over TCP socket.
#          Used by collect_esp.py for training data collection.
#
# Protocol: Client sends text commands ("capture", "capture_resized", "quit")
#           Server responds with: 4-byte image size (big-endian) + raw BMP data
#
# IMPORTANT: Connect WiFi manually in the REPL before running this script.
#
# Author: Shravani Poman (with assistance from Claude AI)
# Course: AI for Engineers, Spring 2026

import socket
import time
from camera import Camera, GrabMode, PixelFormat, FrameSize, GainCeiling
from image_preprocessing import resize_96x96_to_32x32_and_threshold  # SEEED Studio

# --- Camera Pin Configuration for XIAO ESP32-S3 Sense ---
CAMERA_PARAMETERS = {
    "data_pins": [15, 17, 18, 16, 14, 12, 11, 48],  # 8-bit parallel data bus
    "vsync_pin": 38,       # Vertical sync signal
    "href_pin": 47,        # Horizontal reference signal
    "sda_pin": 40,         # I2C data for camera configuration
    "scl_pin": 39,         # I2C clock for camera configuration
    "pclk_pin": 13,        # Pixel clock from camera
    "xclk_pin": 10,        # Master clock output to camera
    "xclk_freq": 20000000, # 20MHz clock frequency
    "powerdown_pin": -1,   # Not connected
    "reset_pin": -1,       # Not connected
    "pixel_format": PixelFormat.GRAYSCALE,  # Force 8-bit grayscale (not 24-bit color)
}

# Initialize camera in grayscale BMP mode
cam = Camera(**CAMERA_PARAMETERS)
cam.init()
cam.set_bmp_out(True)  # Output BMP format with header
print("Camera initialized (grayscale mode)")

# --- WiFi Setup ---
# WiFi must be connected manually in REPL before running:
#   import network
#   wlan = network.WLAN(network.STA_IF)
#   wlan.active(True)
#   wlan.connect("YourSSID", "YourPassword")
import network
wlan = network.WLAN(network.STA_IF)
ip = wlan.ifconfig()[0]  # Get assigned IP address
print(f"ESP32 IP address: {ip}")

# --- TCP Socket Server ---
PORT = 8080  # Port to listen on
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Create TCP socket
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow port reuse
server.bind(('0.0.0.0', PORT))  # Listen on all interfaces
server.listen(1)  # Max 1 pending connection
print(f"Server listening on {ip}:{PORT}")
print("Waiting for client to connect...")

# --- Main Server Loop ---
while True:
    # Block until a client connects
    client, addr = server.accept()
    print(f"Client connected from {addr}")
    try:
        while True:
            # Read text command from client
            request = client.recv(1024).decode().strip()

            if request == "capture":
                # Capture and send full-resolution grayscale BMP (128x128, ~17KB)
                img = cam.capture()
                print(f"Captured: {len(img)} bytes")
                # Send 4-byte size header followed by image data
                client.send(len(img).to_bytes(4, 'big'))
                client.send(img)

            elif request == "capture_resized":
                # Capture and resize to 32x32 on ESP32 using SEEED's function
                img = cam.capture()
                resized = resize_96x96_to_32x32_and_threshold(img, -1)  # -1 = no threshold
                client.send(len(resized).to_bytes(4, 'big'))
                client.send(bytes(resized))

            elif request == "quit":
                # Client requested disconnect
                print("Client disconnecting")
                break

            else:
                print(f"Unknown: {request}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Close client and wait for next connection
        client.close()
        print("Waiting for next connection...")
