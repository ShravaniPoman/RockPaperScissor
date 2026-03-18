# socket_server.py — runs on the ESP32
# Captures images from the camera and sends them over WiFi to your laptop

import socket
import time
from camera import Camera, GrabMode, PixelFormat, FrameSize, GainCeiling
#from Wifi import Sta
from image_preprocessing import resize_96x96_to_32x32_and_threshold

# --- Camera Setup ---
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

cam = Camera(**CAMERA_PARAMETERS)
cam.init()
cam.set_bmp_out(True)
print("Camera initialized")

# --- WiFi Setup ---
# --- WiFi Setup ---
import network
wlan = network.WLAN(network.STA_IF)
ip = wlan.ifconfig()[0]
print(f"ESP32 IP address: {ip}")

# --- Socket Server ---
PORT = 8080
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('0.0.0.0', PORT))
server.listen(1)
print(f"Server listening on {ip}:{PORT}")
print("Waiting for client to connect...")

while True:
    client, addr = server.accept()
    print(f"Client connected from {addr}")

    try:
        while True:
            # Wait for a request from client
            request = client.recv(1024).decode().strip()

            if request == "capture":
                # Capture image
                img = cam.capture()
                print(f"Captured image: {len(img)} bytes")

                # Send the image size first (as 4 bytes)
                img_size = len(img)
                client.send(img_size.to_bytes(4, 'big'))

                # Send the image data
                client.send(img)
                print("Image sent")

            elif request == "capture_resized":
                # Capture and resize to 32x32 with threshold
                img = cam.capture()
                resized = resize_96x96_to_32x32_and_threshold(img, -1)
                print(f"Captured and resized image: {len(resized)} bytes")

                # Send the image size first
                img_size = len(resized)
                client.send(img_size.to_bytes(4, 'big'))

                # Send the resized image data
                client.send(bytes(resized))
                print("Resized image sent")

            elif request == "stream":
                # Continuously stream images
                print("Streaming started...")
                while True:
                    img = cam.capture()
                    img_size = len(img)
                    client.send(img_size.to_bytes(4, 'big'))
                    client.send(img)
                    time.sleep(0.1)

            elif request == "quit":
                print("Client requested disconnect")
                break

            else:
                print(f"Unknown request: {request}")

    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        client.close()
        print("Client disconnected. Waiting for new connection...")
