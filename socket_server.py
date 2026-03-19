# socket_server.py — Runs on the ESP32 (MicroPython)
# WiFi camera server — captures GRAYSCALE images and sends to laptop
#
# Author: Shravani Poman (with assistance from Claude AI)

import socket
import time
from camera import Camera, GrabMode, PixelFormat, FrameSize, GainCeiling
from image_preprocessing import resize_96x96_to_32x32_and_threshold

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
    "pixel_format": PixelFormat.GRAYSCALE,  # Force 8-bit grayscale
}

cam = Camera(**CAMERA_PARAMETERS)
cam.init()
cam.set_bmp_out(True)
print("Camera initialized (grayscale mode)")

import network
wlan = network.WLAN(network.STA_IF)
ip = wlan.ifconfig()[0]
print(f"ESP32 IP address: {ip}")

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
            request = client.recv(1024).decode().strip()
            if request == "capture":
                img = cam.capture()
                print(f"Captured: {len(img)} bytes")
                client.send(len(img).to_bytes(4, 'big'))
                client.send(img)
            elif request == "capture_resized":
                img = cam.capture()
                resized = resize_96x96_to_32x32_and_threshold(img, -1)
                client.send(len(resized).to_bytes(4, 'big'))
                client.send(bytes(resized))
            elif request == "quit":
                print("Client disconnecting")
                break
            else:
                print(f"Unknown: {request}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()
        print("Waiting for next connection...")
