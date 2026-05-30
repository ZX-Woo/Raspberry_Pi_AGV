# Import the OpenCV library for computer vision (drawing boxes, finding shapes)
import cv2
# Import NumPy for doing heavy math and handling data arrays (used for color ranges)
import numpy as np
# Import time (not heavily used right now, but good for adding delays if needed later)
import time
# Import the official Raspberry Pi camera library
from picamera2 import Picamera2

# ==============================================================================
# 1. SETUP & HARDWARE CALIBRATION
# ==============================================================================

print("Initializing Picamera2...")
picam2 = Picamera2()

config = picam2.create_video_configuration(main={"size": (320, 240), "format": "RGB888"})
picam2.configure(config)
picam2.start()

print("Applying hardcoded camera settings...")
picam2.set_controls({"AwbEnable": False, "AeEnable": False, "FrameRate": 30})

# --- CALIBRATION ZONE ---
red_gain = 1.8   
blue_gain = 1.2  
picam2.set_controls({"ColourGains": (red_gain, blue_gain)})

picam2.set_controls({"AnalogueGain": 4.0}) 

print("Camera locked in pure manual mode at 30 FPS!")

# --- Define HSV Color Ranges ---
color_ranges = {
    'Red': [
        (np.array([0, 50, 50]), np.array([4, 255, 255])),
        (np.array([170, 50, 50]), np.array([179, 255, 255]))
    ],
    'Yellow': [(np.array([22, 100, 100]), np.array([35, 255, 255]))],
    'Green':  [(np.array([36, 50, 50]),   np.array([88, 255, 255]))],
    'Cyan':   [(np.array([90, 100, 100]), np.array([108, 255, 255]))],
    'Blue':   [(np.array([110, 40, 30]),  np.array([132, 255, 255]))],
    'Purple': [(np.array([133, 40, 40]),  np.array([168, 255, 255]))],
    'Orange': [(np.array([5, 50, 50]),    np.array([20, 255, 255]))]
}

# ==============================================================================
# 2. BULLETPROOF DETECTION FUNCTIONS
# ==============================================================================

def process_mask(mask, color_name, display_frame, hsv_frame, frame_tally):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 400: continue 
            
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0 or w == 0: continue 
        x, y = max(0, x), max(0, y)
        
        perim = cv2.arcLength(cnt, True)
        if perim == 0: continue
            
        circularity = (4 * np.pi * area) / (perim * perim)
        box_area = w * h
        area_ratio = area / float(box_area)
        
        approx = cv2.approxPolyDP(cnt, 0.04 * perim, True)
        verts = len(approx)
        
        # Bounding Box Center
        bx, by = x + w // 2, y + h // 2
        
        # --- DYNAMIC ARROW DIRECTION MATH (CENTER OF MASS) ---
        # We calculate the true physical center of weight of the shape.
        # The pointy tip of the arrow will always stretch furthest from this center point!
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            mass_cx = int(M["m10"] / M["m00"])
            mass_cy = int(M["m01"] / M["m00"])
        else:
            mass_cx, mass_cy = bx, by # Fallback just in case
            
        # Figure out which way the pointy tip is stretching
        if w > h: # It is a Horizontal Arrow
            dist_left = mass_cx - x
            dist_right = (x + w) - mass_cx
            arrow_dir = "Right" if dist_right > dist_left else "Left"
        else: # It is a Vertical Arrow
            dist_top = mass_cy - y
            dist_bot = (y + h) - mass_cy
            arrow_dir = "Down" if dist_bot > dist_top else "Up"
        
        # Grab color for HUD
        if 0 <= by < hsv_frame.shape[0] and 0 <= bx < hsv_frame.shape[1]:
            hue_value = hsv_frame[by, bx][0]
        else:
            hue_value = 0 
            
        label = "Unknown"
        
        # --- THE FORGIVING LOGIC TREE ---
        if color_name == "Yellow": 
            if 3 <= verts <= 6 and area_ratio > 0.60: label = "Biohazard"
            else: label = "Yellow Star"
            
        elif color_name == "Cyan": 
            if 7 <= verts <= 9: label = "Cyan Octagon" 
            
        elif color_name == "Green": 
            if verts >= 4 and area_ratio > 0.45: label = "Button"
            else: label = f"Green Arrow ({arrow_dir})" 
            
        elif color_name == "Orange":
            if verts == 6: label = f"Orange Arrow ({arrow_dir})"
            else: label = "Orange Cross"
            
        elif color_name == "Blue":
            if circularity > 0.40 or verts > 6: label = "Blue 3/4 Circle" 
            else: label = f"Blue Arrow ({arrow_dir})"
            
        elif color_name == "Red":
            if circularity > 0.50 or verts > 8: label = "Red Inc. Circle"
            else: label = f"Red Arrow ({arrow_dir})"
            
        elif color_name == "Purple":
            if circularity < 0.68: label = "Purple Trapezium"
            elif circularity > 0.68: label = "Purple Diamond"
            else: label = "Purple Unknown"
        
        # --- HUD DRAWING & TALLYING ---
        if label != "Unknown":
            frame_tally[label] = frame_tally.get(label, 0) + 1
            
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            
            display_text = f"{label} [V:{verts}] [C:{circularity:.2f}]"
            cv2.putText(display_frame, display_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            
            # Draw the corners
            for vertex in approx:
                vx, vy = vertex[0]
                cv2.circle(display_frame, (vx, vy), 4, (0, 0, 255), -1)
            # Draw a Blue dot on the Center of Mass so you can see the math working!
            cv2.circle(display_frame, (mass_cx, mass_cy), 4, (255, 0, 0), -1)


def detect_colored_shapes(bgr_frame, display_frame):
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    frame_tally = {} 
    
    for color_name, ranges in color_ranges.items():
        combined_mask = np.zeros(hsv.shape[:2], dtype="uint8")
        
        for (lower, upper) in ranges:
            mask_part = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask_part)
            
        process_mask(combined_mask, color_name, display_frame, hsv, frame_tally)
        
    return frame_tally

# ==============================================================================
# 3. MAIN LOOP & COMBO LOGIC
# ==============================================================================

print("Starting Main Detection Loop...")

try:
    while True:
        raw_frame = picam2.capture_array()
        frame = raw_frame
        
        tally = detect_colored_shapes(frame, frame)
        
        # --- COMBO COUNTING ---
        purple_count = tally.get("Purple Diamond", 0) + tally.get("Purple Trapezium", 0)
        blue_circle_count = tally.get("Blue 3/4 Circle", 0)
        biohazard_count = tally.get("Biohazard", 0)
        button_count = tally.get("Button", 0)
        
        # DYNAMIC EXTRACTION: Search the dictionary for arrows, no matter the direction
        green_arrow_count = sum(count for name, count in tally.items() if "Green Arrow" in name)
        
        # DYNAMIC ARROW TRACKER: If any arrow is found, pull its exact direction out of the text
        arrow_detected = None
        for name in tally.keys():
            if "Arrow" in name:
                # This mathematically rips the "(Up)" or "(Left)" right out of the label!
                arrow_detected = name.split("(")[1].strip(")")
                break 
        
        symbol_detected = None
        
        # --- THE MASTER COMBO RULES ---
        if biohazard_count >= 1: symbol_detected = "BIOHAZARD"
        elif button_count >= 1: symbol_detected = "BUTTON"
        elif purple_count >= 2: symbol_detected = "FINGERPRINT"
        elif blue_circle_count >= 3: symbol_detected = "QR CODE"
        elif green_arrow_count >= 3: symbol_detected = "RECYCLE"
        # THE DYNAMIC RULE: Will print ARROW LEFT, ARROW UP, etc. dynamically!
        elif arrow_detected: symbol_detected = f"ARROW {arrow_detected.upper()}"
            
        if symbol_detected:
            cv2.putText(frame, f"*** SYMBOL: {symbol_detected} ***", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
        cv2.imshow("AGV Vision HUD", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

except KeyboardInterrupt:
    print("\nProgram stopped by user.")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera safely closed.")