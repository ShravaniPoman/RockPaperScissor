# web_stream_fast.py — Runs on the ESP32 (MicroPython)
# Purpose: HTTP web server that streams camera images to a web browser.
#          Supports full-size (128x128) and 32x32 CNN-ready image modes.
#          HTML is sent as small byte chunks to avoid ESP32 memory limits.
#
# Usage: Connect WiFi, then run this script. Open the printed URL in a browser.
#
# Author: Shravani Poman (with assistance from Claude AI)


import socket
import network
from time import sleep
from camera import Camera, GrabMode, PixelFormat, FrameSize, GainCeiling
from image_preprocessing import resize_96x96_to_32x32_and_threshold  # SEEED Studio

# --- Camera Pin Configuration (same as socket_server.py) ---
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
print("Camera initialized")

# --- WiFi Connection ---
# Connects to phone hotspot. Update SSID and password for your network.
wlan = network.WLAN(network.STA_IF)
if not wlan.isconnected():
    network.WLAN(network.AP_IF).active(False)  # Disable access point mode
    sleep(1)
    wlan.active(True)  # Enable station (client) mode
    wlan.connect("Shravani's iPhone", "******")  # UPDATE with your credentials
    while not wlan.isconnected():
        print("Waiting...")
        sleep(2)

ip = wlan.ifconfig()[0]  # Get assigned IP address
print(f"ESP32 IP address: {ip}")

def send_page(cl):
    """
    Sends the HTML page to the browser client in small byte chunks.
    We use b"..." byte strings instead of one large string to avoid
    MicroPython memory errors on the ESP32 with large string allocations.
    
    The page includes:
    - Camera image display (scaled up from 32x32 to 384x384 for visibility)
    - Mode toggle buttons (Full Size vs 32x32 CNN mode)
    - Capture/Stream/Stop buttons
    - FPS counter
    - JavaScript that fetches images from /capture or /capture_small endpoints
    """
    # HTTP response header
    cl.send(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n")
    # HTML head with CSS styling
    cl.send(b"<html><head><title>ESP32 Cam</title>")
    cl.send(b"<style>body{background:#1a1a2e;color:#fff;text-align:center;font-family:Arial;}")
    cl.send(b"img{border:3px solid #e94560;border-radius:10px;margin:20px;image-rendering:pixelated;}")
    cl.send(b"button{background:#e94560;color:#fff;border:none;padding:12px 24px;margin:5px;")
    cl.send(b"border-radius:5px;font-size:16px;cursor:pointer;}")
    cl.send(b"h1{color:#e94560;}.active{background:#27ae60;}</style></head>")
    # HTML body with image display and control buttons
    cl.send(b"<body><h1>ESP32 Camera Stream</h1>")
    cl.send(b"<div><img id='s' src='/capture_small' width='384' height='384'></div>")
    # Mode selection buttons - switch between full size and 32x32 CNN view
    cl.send(b"<div><button onclick=\"M('full')\" id='bf'>Full Size</button>")
    cl.send(b"<button onclick=\"M('small')\" id='bs' class='active'>32x32 CNN</button></div>")
    # Action buttons - single capture, start/stop streaming
    cl.send(b"<div><button onclick='C()'>Capture</button>")
    cl.send(b"<button onclick='G()'>Start Stream</button>")
    cl.send(b"<button onclick='S()'>Stop Stream</button></div>")
    cl.send(b"<div id='f' style='color:#aaa'>FPS: --</div>")
    # JavaScript for image fetching and streaming logic
    # M(v) = set mode (full/small), U() = get URL for current mode
    # C() = capture single frame, G() = start streaming, S() = stop streaming
    # L() = streaming loop - loads next image only after previous one finishes (prevents pile-up)
    cl.send(b"<script>var r=false,m='small',fc=0,lt=Date.now();")
    cl.send(b"function M(v){m=v;document.getElementById('bf').className=v=='full'?'active':'';")
    cl.send(b"document.getElementById('bs').className=v=='small'?'active':'';C();}")
    cl.send(b"function U(){return m=='small'?'/capture_small?':'/capture?';}")
    cl.send(b"function C(){document.getElementById('s').src=U()+Date.now();}")
    cl.send(b"function G(){r=true;fc=0;lt=Date.now();L();}")
    cl.send(b"function L(){if(!r)return;var i=new Image();i.onload=function(){")
    cl.send(b"document.getElementById('s').src=i.src;fc++;var n=Date.now();")
    # FPS counter: counts frames per second and updates display
    cl.send(b"if(n-lt>=1000){document.getElementById('f').innerText='FPS: '+fc;fc=0;lt=n;}")
    cl.send(b"if(r)L();};i.src=U()+Date.now();}")
    cl.send(b"function S(){r=false;}")
    cl.send(b"</script></body></html>")

# --- HTTP Web Server Setup ---
PORT = 80  # Standard HTTP port
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow port reuse
server.bind(('0.0.0.0', PORT))  # Listen on all interfaces
server.listen(5)  # Allow up to 5 pending connections (browser makes multiple requests)
print(f"Web server running at http://{ip}")

# --- Main Request Handler Loop ---
while True:
    try:
        # Accept incoming HTTP request from browser
        client, addr = server.accept()
        request = client.recv(1024).decode()  # Read the HTTP request line

        if "GET /capture_small" in request:
            # Serve a 32x32 resized grayscale BMP image (CNN-ready)
            # Must check /capture_small BEFORE /capture since it contains "capture"
            img = cam.capture()
            resized = resize_96x96_to_32x32_and_threshold(img, -1)  # -1 = no threshold
            # Send HTTP response with BMP content type and correct content length
            h = "HTTP/1.1 200 OK\r\nContent-Type: image/bmp\r\nContent-Length: {}\r\nConnection: close\r\n\r\n".format(len(resized))
            client.send(h.encode())
            client.send(bytes(resized))

        elif "GET /capture" in request:
            # Serve full-resolution 128x128 grayscale BMP image
            img = cam.capture()
            h = "HTTP/1.1 200 OK\r\nContent-Type: image/bmp\r\nContent-Length: {}\r\nConnection: close\r\n\r\n".format(len(img))
            client.send(h.encode())
            client.send(img)

        elif "GET / " in request or "GET / HTTP" in request:
            # Serve the main HTML page with camera viewer UI
            send_page(client)

        else:
            # Return 404 for any other requests (favicon.ico, etc.)
            client.send(b"HTTP/1.1 404 Not Found\r\n\r\n")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Always close the connection (HTTP/1.0 style - one request per connection)
        client.close()
