import cv2
import numpy as np
# Import the official Raspberry Pi Camera library
from picamera2 import Picamera2

# ==============================================================================
# 1. SETUP & INITIALIZATION
# ==============================================================================

print("Initializing Picamera2...")
# Create the camera object
picam2 = Picamera2()

# Configure the camera: 320x240 resolution, and BGR format (which OpenCV requires)
config = picam2.create_video_configuration(main={"size": (320, 240), "format": "RGB888"})
picam2.configure(config)

# Start the camera hardware
picam2.start()
print("Camera started successfully!")

# Load Symbol Templates (Make sure these are small! e.g., 50x50 pixels)
templates = {
    'Biohazard': cv2.imread('biohazard.png', 0),
    'Stop': cv2.imread('stop_sign.png', 0)
}

# Define HSV Color Ranges (Hue, Saturation, Value)
color_ranges = {
    'Yellow': (np.array([20, 100, 100]), np.array([35, 255, 255])),
    'Green':  (np.array([40, 50, 50]),   np.array([80, 255, 255])),
    'Blue':   (np.array([100, 50, 50]),  np.array([130, 255, 255])),
    'Purple': (np.array([130, 50, 50]),  np.array([160, 255, 255]))
}

# ==============================================================================
# 2. DETECTION FUNCTIONS
# ==============================================================================
def detect_colors(bgr_frame, display_frame):
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    for color_name, (lower_bound, upper_bound) in color_ranges.items():
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) > 300: 
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(display_frame, f"{color_name} Obj", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

def detect_geometry_and_arrows(gray_frame, display_frame):
    blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150) 
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv2.contourArea(cnt) < 150: 
            continue 
        perim = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perim, True)
        verts = len(approx)
        x, y, w, h = cv2.boundingRect(approx)
        
        if verts == 3: label = "Triangle"
        elif verts == 4: label = "Square/Rect"
        elif verts == 7: label = "ARROW"
        else: continue 
            
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(display_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def run_template_matching(gray_frame, display_frame):
    for name, template in templates.items():
        if template is None: continue
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val >= 0.75: 
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(display_frame, top_left, bottom_right, (255, 0, 0), 2)
            cv2.putText(display_frame, f"{name} ({int(max_val*100)}%)", 
                        (top_left[0], top_left[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            return

# ==============================================================================
# 3. MAIN ROBOT VISION LOOP
# ==============================================================================

print("Starting Robot Vision System (Picamera2 Backend)...")
frame_counter = 0 

try:
    while True:
        # THE NEW WAY TO GET A FRAME:
        # This grabs the frame directly from the hardware as a NumPy array
        frame = picam2.capture_array()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_counter += 1
        
        detect_colors(frame, frame)
        detect_geometry_and_arrows(gray, frame)
        
        if frame_counter % 5 == 0:
            run_template_matching(gray, frame)
            
        cv2.imshow("Robot Vision HUD", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

except KeyboardInterrupt:
    print("Program stopped by user.")

finally:
    # Safely shut down the camera when the script ends
    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera safely released.")