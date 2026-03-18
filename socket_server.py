# socket_server.py — Runs on the ESP32 (MicroPython)
# Purpose: WiFi camera server that captures images from the OV3660 camera
#          and sends them over TCP to a laptop client for data collection
#          or real-time classification.
#
# Protocol: Client sends text commands ("capture", "capture_resized", "stream", "quit")
#           Server responds with: 4-byte image size (big-endian) + raw BMP image data
#
# Usage: Connect WiFi manually in REPL first, then run this script.
#
# Author: Shravani Poman (with assistance from Claude AI)


import socket
import time
from camera import Camera, GrabMode, PixelFormat, FrameSize, GainCeiling
from image_preprocessing import resize_96x96_to_32x32_and_threshold  # SEEED Studio preprocessing

# --- Camera Pin Configuration for XIAO ESP32-S3 Sense ---
# These pin assignments match the hardware wiring between the ESP32-S3
# processor and the OV3660 camera module on the XIAO Sense board
CAMERA_PARAMETERS = {
    "data_pins": [15, 17, 18, 16, 14, 12, 11, 48],  # 8-bit parallel data bus from camera
    "vsync_pin": 38,       # Vertical sync - signals start of new frame
    "href_pin": 47,        # Horizontal reference - signals valid pixel data in row
    "sda_pin": 40,         # I2C data line for configuring camera registers
    "scl_pin": 39,         # I2C clock line for configuring camera registers
    "pclk_pin": 13,        # Pixel clock - camera outputs one pixel per clock cycle
    "xclk_pin": 10,        # Master clock - ESP32 provides clock signal to camera
    "xclk_freq": 20000000, # 20MHz master clock frequency
    "powerdown_pin": -1,   # Not connected on this board (-1 = disabled)
    "reset_pin": -1,       # Not connected on this board (-1 = disabled)
}

# Initialize the camera hardware and set output format
cam = Camera(**CAMERA_PARAMETERS)
cam.init()  # Powers on camera and configures registers via I2C
cam.set_bmp_out(True)  # Output 8-bit grayscale BMP format (not compressed JPEG)
print("Camera initialized")

# --- WiFi Setup ---
# IMPORTANT: WiFi must be connected manually in the REPL before running this script.
# The Wifi.py helper module is unreliable and throws "Wifi Internal Error".
# Manual connection steps:
#   import network
#   wlan = network.WLAN(network.STA_IF)
#   wlan.active(True)
#   wlan.connect("YourSSID", "YourPassword")
#   # Wait until wlan.isconnected() returns True
import network
wlan = network.WLAN(network.STA_IF)
ip = wlan.ifconfig()[0]  # Get the IP address assigned by the WiFi network
print(f"ESP32 IP address: {ip}")

# --- TCP Socket Server Setup ---
PORT = 8080  # Port number the server listens on
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Create TCP socket
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow reuse after restart
server.bind(('0.0.0.0', PORT))  # Bind to all network interfaces on this port
server.listen(1)  # Queue up to 1 pending connection
print(f"Server listening on {ip}:{PORT}")
print("Waiting for client to connect...")

# --- Main Server Loop ---
# Continuously accepts client connections and processes their requests
while True:
    # server.accept() blocks here until a client connects
    client, addr = server.accept()
    print(f"Client connected from {addr}")

    try:
        # Inner loop: handle multiple requests from the same client
        while True:
            # Receive text command from client (e.g. "capture", "quit")
            request = client.recv(1024).decode().strip()

            if request == "capture":
                # Capture a full-resolution image from the camera
                # Returns a 128x128 pixel 8-bit grayscale BMP (~49KB)
                img = cam.capture()
                print(f"Captured image: {len(img)} bytes")

                # Send the image size as a 4-byte big-endian integer first
                # This tells the client exactly how many bytes to expect
                img_size = len(img)
                client.send(img_size.to_bytes(4, 'big'))

                # Send the raw BMP image data
                client.send(img)
                print("Image sent")

            elif request == "capture_resized":
                # Capture and resize to 32x32 pixels on the ESP32
                # Uses SEEED's preprocessing function from image_preprocessing.py
                # threshold=-1 means keep full grayscale (no binary black/white conversion)
                img = cam.capture()
                resized = resize_96x96_to_32x32_and_threshold(img, -1)
                print(f"Captured and resized image: {len(resized)} bytes")

                # Send size header + resized BMP data
                img_size = len(resized)
                client.send(img_size.to_bytes(4, 'big'))
                client.send(bytes(resized))
                print("Resized image sent")

            elif request == "stream":
                # Continuous streaming mode - sends frames as fast as possible
                # Used for live preview in browser or continuous capture
                print("Streaming started...")
                while True:
                    img = cam.capture()
                    img_size = len(img)
                    client.send(img_size.to_bytes(4, 'big'))
                    client.send(img)
                    time.sleep(0.1)  # Brief delay to target ~10 FPS

            elif request == "quit":
                # Client requested clean disconnect
                print("Client requested disconnect")
                break

            else:
                print(f"Unknown request: {request}")

    except Exception as e:
        # Handle network errors (client disconnect, timeout, etc.)
        print(f"Connection error: {e}")
    finally:
        # Always close the client socket and loop back to accept new connections
        client.close()
        print("Client disconnected. Waiting for new connection...")
