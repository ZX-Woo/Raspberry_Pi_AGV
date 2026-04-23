# Import the OpenCV library for computer vision
import cv2
import numpy as np
import time
import math
from picamera2 import Picamera2

# ==============================================================================
# 1. SETUP & HARDWARE CALIBRATION
# ==============================================================================

print("Initializing Picamera2...")
picam2 = Picamera2()

config = picam2.create_video_configuration(main={"size": (320, 240), "format": "BGR888"})
picam2.configure(config)
picam2.start()

print("Applying hardcoded camera settings...")
picam2.set_controls({
    "AwbEnable": False, 
    "AeEnable": False, 
    "FrameRate": 30,
    "ExposureTime": 22000, 
    "AnalogueGain": 4.0,
    "ColourGains": (1.8, 1.2)
})

print("Camera locked! Running Restored Geometry & Fatigue-Proof Combo Mode.")

kernel = np.ones((5, 5), np.uint8)

# ==============================================================================
# 2. THE PURE GEOMETRY MATH
# ==============================================================================

def process_shapes(mask, bgr_frame, display_frame, hsv_frame, frame_tally):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        
        # --- SHADOW & DUST KILLER ---
        if area < 400 or area > 12000: continue 
            
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0 or w == 0: continue 
            
        if w > 200 or h > 200: continue
            
        x, y = max(0, x), max(0, y)
        perim = cv2.arcLength(cnt, True)
        if perim == 0: continue
            
        box_area = w * h
        area_ratio = area / float(box_area)
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        
        hull_ratio = hull_area / float(box_area)
        
        # THE STIFF RULER: 0.04 forces wobbly fingerprint ridges to snap back to 4 corners!
        epsilon = 0.04 * perim
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        verts = len(approx)
        
        is_convex = cv2.isContourConvex(approx)
        
        # --- COUNTING THE DEEP DENTS (Defects) ---
        defect_count = 0
        try:
            hull_pts_idx = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull_pts_idx)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    if d > 2000: 
                        defect_count += 1
        except:
            defect_count = 0
            
        # --- DYNAMIC ARROW DIRECTION ---
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2
            
        max_dist = 0
        tip_x, tip_y = cx, cy
        for pt in cnt:
            px, py = pt[0][0], pt[0][1]
            dist = (px - cx)**2 + (py - cy)**2
            if dist > max_dist:
                max_dist = dist
                tip_x, tip_y = px, py
                
        dx = tip_x - cx
        dy = tip_y - cy
        
        if abs(dx) > abs(dy): arrow_dir = "Right" if dx > 0 else "Left"
        else: arrow_dir = "Down" if dy > 0 else "Up"
            
        # ======================================================================
        # THE FLAWLESS LOGIC TREE (ALL NAMES RESTORED TO PREVENT CONFUSION)
        # ======================================================================
        label = "Unknown"
        
        # --- TOP-LEFT SNIFFER (For Button & Biohazard) ---
        test_x, test_y = x + (w // 6), y + (h // 6)
        if 0 <= test_y < hsv_frame.shape[0] and 0 <= test_x < hsv_frame.shape[1]:
            hue = hsv_frame[test_y, test_x][0]
        else: hue = 0
        
        # 1. THE EMPTY SHAPES (Arrow vs Star)
        if area_ratio <= 0.45:
            if defect_count >= 5 or verts >= 10: label = "Star"
            else: label = f"Arrow ({arrow_dir})"
            
        # 2. THE CHUNKY BLOCKS (Button, Biohazard, QR Squares)
        elif hull_ratio > 0.85:
            if area > 1500:
                if 10 <= hue <= 35: label = "Biohazard"
                else: label = "Button" 
                cv2.circle(display_frame, (test_x, test_y), 4, (0, 255, 255), -1)
            else:
                label = "QR Square"
                
        # 3. STRICTLY 4 CORNERS (Diamond & Trapezium)
        elif verts == 4:
            if area_ratio >= 0.55: label = "Trapezium"
            else: label = "Diamond"
            
        # 4. COUNTING THE DENTS (3/4 Circle & Cross)
        elif defect_count == 1:
            label = "3/4 Circle" 
            
        elif defect_count == 4:
            label = "Cross" 
            
        # 5. CONVEX SHAPES (No Dents)
        elif is_convex or defect_count == 0:
            if verts == 8: label = "Octagon"
            elif verts <= 7: label = "Semicircle"
            else: label = "Circle"
        
        # --- HUD DRAWING ---
        if label != "Unknown":
            frame_tally[label] = frame_tally.get(label, 0) + 1
            
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            display_text = f"{label} [V:{verts}] [A:{area_ratio:.2f}]"
            cv2.putText(display_frame, display_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 2)
            
            for vertex in approx:
                vx, vy = vertex[0]
                cv2.circle(display_frame, (vx, vy), 4, (0, 0, 255), -1)
                
            if "Arrow" in label:
                cv2.circle(display_frame, (cx, cy), 4, (255, 0, 0), -1)
                cv2.circle(display_frame, (tip_x, tip_y), 5, (0, 255, 255), -1)

def detect_shapes(bgr_frame, display_frame):
    yuv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YUV)
    y_channel = yuv[:, :, 0] 
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    
    blurred_y = cv2.GaussianBlur(y_channel, (5, 5), 0)
    _, bw_mask = cv2.threshold(blurred_y, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    bw_mask = cv2.morphologyEx(bw_mask, cv2.MORPH_OPEN, kernel)
    bw_mask = cv2.morphologyEx(bw_mask, cv2.MORPH_CLOSE, kernel)
    
    frame_tally = {} 
    process_shapes(bw_mask, bgr_frame, display_frame, hsv, frame_tally)
        
    return frame_tally, bw_mask

# ==============================================================================
# 3. MAIN LOOP & COMBO LOGIC
# ==============================================================================

print("Starting Main Detection Loop...")

try:
    while True:
        raw_frame = picam2.capture_array()
        frame = raw_frame
        
        tally, bw_screen = detect_shapes(frame, frame)
        
        # --- COMBO COUNTING ---
        # Separated so we can cheat the Fingerprint logic
        perfect_fingerprint = tally.get("Diamond", 0) + tally.get("Trapezium", 0)
        wobbly_fingerprint = tally.get("Star", 0) + tally.get("Semicircle", 0)
        
        biohazard_count = tally.get("Biohazard", 0)
        button_count = tally.get("Button", 0)
        qr_square_count = tally.get("QR Square", 0)
        
        arrow_count = sum(count for name, count in tally.items() if "Arrow" in name)
        
        arrow_detected = None
        for name in tally.keys():
            if "Arrow" in name:
                arrow_detected = name.split("(")[1].strip(")")
                break 
        
        symbol_detected = None
        
        # --- THE MASTER COMBO RULES (FATIGUE-PROOF FIX) ---
        if biohazard_count >= 1: symbol_detected = "BIOHAZARD"
        elif button_count >= 1: symbol_detected = "BUTTON"
        elif qr_square_count >= 3: symbol_detected = "QR CODE" 
        elif arrow_count >= 2: symbol_detected = "RECYCLE" 
        
        # THE FINGERPRINT SHORTCUT:
        # If we see even ONE Diamond or Trapezium, force it to FINGERPRINT instantly!
        elif perfect_fingerprint >= 1: symbol_detected = "FINGERPRINT"
        # If both pieces are wobbly, require 2 pieces so a random Star doesn't trigger it.
        elif wobbly_fingerprint >= 2: symbol_detected = "FINGERPRINT"
        
        elif arrow_detected: symbol_detected = f"ARROW {arrow_detected.upper()}"
            
        if symbol_detected:
            cv2.putText(frame, f"*** SYMBOL: {symbol_detected} ***", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
        cv2.imshow("AGV RGB Vision Results", frame)
        cv2.imshow("YUV Pure Luma Mask", bw_screen)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

except KeyboardInterrupt:
    print("\nProgram stopped by user.")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera safely closed.")