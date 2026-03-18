
from camera import Camera, GrabMode, PixelFormat, FrameSize, GainCeiling

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


# Collect all attributes of the cam object that start with "get"
get_methods = [method for method in dir(cam) if callable(getattr(cam, method)) and method.startswith("get")]

# Initialize a dictionary to store the results
results = {}

# Iterate through each "get" method, call it, and store the result
for method in get_methods:
    try:
        # Dynamically call the method and store its result
        result = getattr(cam, method)()
        results[method] = result
    except Exception as e:
        # Handle any errors gracefully
        results[method] = f"Error: {e}"

# Print the results in a readable format
print("Camera 'get' Method Outputs:")
for method, result in results.items():
    print(f"{method}: {result}")
cap = cam.capture()
print(f"Captured Image of size: {len(cap)}")
