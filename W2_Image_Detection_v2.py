import cv2
import numpy as np
import time
import os
from picamera2 import Picamera2

# ==============================================================================
# 1. SETUP & HARDWARE CALIBRATION
# ==============================================================================

print("Initializing Picamera2...")
picam2 = Picamera2()

# By right Capture in RGB so we get raw, true colors.
# At here BGR only get for the true colors.
config = picam2.create_video_configuration(main={"size": (320, 240), "format": "BGR888"})
picam2.configure(config)
picam2.start()

print("Applying hardcoded camera settings...")
picam2.set_controls({"AwbEnable": False, "AeEnable": False, "FrameRate": 30})

# --- CALIBRATION ZONE ---
red_gain = 1.5   
blue_gain = 1.2  
picam2.set_controls({"ColourGains": (red_gain, blue_gain)})
picam2.set_controls({"AnalogueGain": 3.0}) 
# ------------------------

print("Camera locked in pure manual mode at 30 FPS!")

# --- Load 50x50 Templates ---
base_path = os.path.expanduser('~/Downloads/Symbols/')
templates = {
    'Button':      cv2.imread(os.path.join(base_path, 'Button.png'), 0),
    'Biohazard':   cv2.imread(os.path.join(base_path, 'Biohazard.png'), 0),
    'Recycle':     cv2.imread(os.path.join(base_path, 'Recycle .png'), 0),
    'Fingerprint': cv2.imread(os.path.join(base_path, 'Fingerprint .png'), 0),
    'QR Code':     cv2.imread(os.path.join(base_path, 'QR Code.png'), 0)
}

# Safety Check: Warn you immediately if a symbol didn't load!
for name, img in templates.items():
    if img is None:
        print(f"CRITICAL WARNING: '{name}' template failed to load! Check your folder path.")

# --- Define HSV Color Ranges ---
color_ranges = {
    'Yellow': (np.array([15, 100, 100]), np.array([35, 255, 255])),
    'Green':  (np.array([40, 50, 50]),   np.array([85, 255, 255])),
    'Cyan':   (np.array([85, 100, 100]), np.array([105, 255, 255])),
    'Blue':   (np.array([105, 100, 50]), np.array([130, 255, 255])),
    'Purple': (np.array([130, 50, 50]),  np.array([160, 255, 255])),
    'Orange': (np.array([5, 100, 100]),  np.array([15, 255, 255]))
}

# ==============================================================================
# 2. BULLETPROOF DETECTION FUNCTIONS
# ==============================================================================

def process_mask(mask, color_name, display_frame, hsv_frame, gray_frame):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 400: # Filter noise
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        
        # SAFETY CHECK: Prevent divide-by-zero crash if height is 0
        if h == 0: continue 
        
        aspect_ratio_box = w / float(h) 
        
        # =========================================================
        # --- PRIORITY 1: IS IT A SYMBOL? (ANTI-SQUISH MATCHING) ---
        # =========================================================
        roi = gray_frame[y:y+h, x:x+w]
        
        if 0.8 <= aspect_ratio_box <= 1.2 and roi.shape[0] > 20 and roi.shape[1] > 20: 
            roi_resized = cv2.resize(roi, (50, 50))
            
            best_match_val = 0
            best_match_name = ""
            
            for name, template in templates.items():
                if template is None: continue
                res = cv2.matchTemplate(roi_resized, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                
                if max_val > best_match_val:
                    best_match_val = max_val
                    best_match_name = name
            
            if best_match_val > 0.50:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 255), 3)
                cv2.putText(display_frame, f"Sym: {best_match_name} ({int(best_match_val*100)}%)", 
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                continue 

        # =========================================================
        # --- PRIORITY 2: IF NOT A SYMBOL, WHAT SHAPE IS IT? ---
        # =========================================================
        perim = cv2.arcLength(cnt, True)
        if perim == 0: continue
            
        circularity = (4 * np.pi * area) / (perim * perim)
        box_area = w * h
        area_ratio = area / float(box_area)
        
        approx = cv2.approxPolyDP(cnt, 0.04 * perim, True)
        verts = len(approx)
        
        cx, cy = x + w // 2, y + h // 2
        hue_value = hsv_frame[cy, cx][0] if (0 <= cy < hsv_frame.shape[0] and 0 <= cx < hsv_frame.shape[1]) else 0
            
        label = "Unknown"
        
        if color_name == "Yellow": label = "Yellow Star"
        elif color_name == "Cyan": 
            if 7 <= verts <= 9: label = "Cyan Octagon" 
        elif color_name == "Green": label = "Green Arrow"
        elif color_name == "Orange":
            # Arrow showed 6 verts and Cross is unstable with a 7 and above verts.
            if verts >= 7: label = "Orange Cross"
            else: label = "Orange Arrow"
        elif color_name == "Blue":
            if circularity > 0.50 or verts > 8: label = "Blue 3/4 Circle" 
            else: label = "Blue Arrow"
        elif color_name == "Red":
            if circularity > 0.50 or verts > 8: label = "Red Inc. Circle"
            else: label = "Red Arrow"
        elif color_name == "Purple":
            if area_ratio < 0.60: label = "Purple Diamond"
            else: label = "Purple Trapezium"
        
        # --- HUD DRAWING ---
        if label != "Unknown":
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.putText(display_frame, f"{label} [V:{verts}] [A:{area_ratio:.2f}] [H:{hue_value}]", 
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            for vertex in approx:
                vx, vy = vertex[0]
                cv2.circle(display_frame, (vx, vy), 4, (0, 0, 255), -1)
            cv2.circle(display_frame, (cx, cy), 4, (255, 0, 0), -1)


def detect_colored_shapes(bgr_frame, display_frame, gray_frame):
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 100, 100]), np.array([5, 255, 255])), 
        cv2.inRange(hsv, np.array([160, 100, 100]), np.array([179, 255, 255]))
    )
    process_mask(mask_red, "Red", display_frame, hsv, gray_frame)
    
    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, lower, upper)
        process_mask(mask, color_name, display_frame, hsv, gray_frame)

# ==============================================================================
# 3. MAIN LOOP
# ==============================================================================

print("Starting Main Detection Loop...")

try:
    while True:
        raw_frame = picam2.capture_array()
        
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        detect_colored_shapes(frame, frame, gray)
            
        cv2.imshow("AGV Vision HUD", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

except KeyboardInterrupt:
    print("\nProgram stopped by user.")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera safely closed.")