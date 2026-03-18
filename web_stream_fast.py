import socket
import network
from time import sleep
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
}

cam = Camera(**CAMERA_PARAMETERS)
cam.init()
cam.set_bmp_out(True)
print("Camera initialized")

wlan = network.WLAN(network.STA_IF)
if not wlan.isconnected():
    network.WLAN(network.AP_IF).active(False)
    sleep(1)
    wlan.active(True)
    wlan.connect("Shravani's iPhone", "pune12345")
    while not wlan.isconnected():
        print("Waiting...")
        sleep(2)

ip = wlan.ifconfig()[0]
print(f"ESP32 IP address: {ip}")

def send_page(cl):
    cl.send(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n")
    cl.send(b"<html><head><title>ESP32 Cam</title>")
    cl.send(b"<style>body{background:#1a1a2e;color:#fff;text-align:center;font-family:Arial;}")
    cl.send(b"img{border:3px solid #e94560;border-radius:10px;margin:20px;image-rendering:pixelated;}")
    cl.send(b"button{background:#e94560;color:#fff;border:none;padding:12px 24px;margin:5px;")
    cl.send(b"border-radius:5px;font-size:16px;cursor:pointer;}")
    cl.send(b"h1{color:#e94560;}.active{background:#27ae60;}</style></head>")
    cl.send(b"<body><h1>ESP32 Camera Stream</h1>")
    cl.send(b"<div><img id='s' src='/capture_small' width='384' height='384'></div>")
    cl.send(b"<div><button onclick=\"M('full')\" id='bf'>Full Size</button>")
    cl.send(b"<button onclick=\"M('small')\" id='bs' class='active'>32x32 CNN</button></div>")
    cl.send(b"<div><button onclick='C()'>Capture</button>")
    cl.send(b"<button onclick='G()'>Start Stream</button>")
    cl.send(b"<button onclick='S()'>Stop Stream</button></div>")
    cl.send(b"<div id='f' style='color:#aaa'>FPS: --</div>")
    cl.send(b"<script>var r=false,m='small',fc=0,lt=Date.now();")
    cl.send(b"function M(v){m=v;document.getElementById('bf').className=v=='full'?'active':'';")
    cl.send(b"document.getElementById('bs').className=v=='small'?'active':'';C();}")
    cl.send(b"function U(){return m=='small'?'/capture_small?':'/capture?';}")
    cl.send(b"function C(){document.getElementById('s').src=U()+Date.now();}")
    cl.send(b"function G(){r=true;fc=0;lt=Date.now();L();}")
    cl.send(b"function L(){if(!r)return;var i=new Image();i.onload=function(){")
    cl.send(b"document.getElementById('s').src=i.src;fc++;var n=Date.now();")
    cl.send(b"if(n-lt>=1000){document.getElementById('f').innerText='FPS: '+fc;fc=0;lt=n;}")
    cl.send(b"if(r)L();};i.src=U()+Date.now();}")
    cl.send(b"function S(){r=false;}")
    cl.send(b"</script></body></html>")

PORT = 80
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('0.0.0.0', PORT))
server.listen(5)
print(f"Web server running at http://{ip}")

while True:
    try:
        client, addr = server.accept()
        request = client.recv(1024).decode()

        if "GET /capture_small" in request:
            img = cam.capture()
            resized = resize_96x96_to_32x32_and_threshold(img, -1)
            h = "HTTP/1.1 200 OK\r\nContent-Type: image/bmp\r\nContent-Length: {}\r\nConnection: close\r\n\r\n".format(len(resized))
            client.send(h.encode())
            client.send(bytes(resized))

        elif "GET /capture" in request:
            img = cam.capture()
            h = "HTTP/1.1 200 OK\r\nContent-Type: image/bmp\r\nContent-Length: {}\r\nConnection: close\r\n\r\n".format(len(img))
            client.send(h.encode())
            client.send(img)

        elif "GET / " in request or "GET / HTTP" in request:
            send_page(client)

        else:
            client.send(b"HTTP/1.1 404 Not Found\r\n\r\n")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()
