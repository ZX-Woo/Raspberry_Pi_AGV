# ==============================================================================
# AGV COMBINED SYSTEM v8 — The Intersection Push
# ==============================================================================
#
# CRITICAL FIXES APPLIED:
#   1. The Intersection Push: Added a 100ms (0.1s) forward drive delay to the 
#      Arrow execution. This accounts for the camera offset and ensures the robot 
#      wheels roll perfectly into the center of the intersection before spinning.
#
# ==============================================================================

import cv2
import numpy as np
import os
import time
import threading
import math
from picamera2 import Picamera2
import pigpio
from flask import Flask, Response, render_template_string

# ==============================================================================
# 1. SHARED STATE
# ==============================================================================
shared_lock = threading.Lock()

shared = {
    "frame":           None,
    "running":         True,
    "main_display":    None,
    "pid_display":     None,
    "vision_display":  None,

    "symbol":          None,
    "symbol_action":   False,
    "cooldown_until":  0.0,

    "current_action":  "IDLE",
    "tracking_label":  "BLACK",
    "on_shortcut":     False,
    "pid_error":       0,

    "key_press":       None,
    "orb_partial_match": None,   
    "orb_inliers":     0, 
    
    "last_arrow_dir":  None,  
}

# ==============================================================================
# 2. CAMERA SETUP & TRIGGER ZONE CONFIG
# ==============================================================================
print("[INIT] Starting camera 320×240 @ 60 FPS...")
picam2 = Picamera2()

camera_config = picam2.create_video_configuration(
    main={"size": (320, 240), "format": "BGR888"},
    controls={"FrameRate": 60}
)
picam2.configure(camera_config)
picam2.start()

picam2.set_controls({
    "AwbEnable":   False,
    "AeEnable":    True,
    "FrameRate":   60,
    "ColourGains": (1.5, 1.2),
})
print("[INIT] Camera ready.")

# --- THE EXPANDED TRIGGER ZONE ---
VISION_ROI_TOP    = 0
VISION_ROI_BOTTOM = 140

TRIGGER_ZONE_X1 = 70    
TRIGGER_ZONE_X2 = 250   
TRIGGER_ZONE_Y1 = 0    
TRIGGER_ZONE_Y2 = 140   

# ==============================================================================
# 3. MOTOR SETUP
# ==============================================================================
ENL, IN1, IN2 = 13, 5, 6
ENR, IN3, IN4 = 12, 19, 26

pi = pigpio.pi()
for pin in [IN1, IN2, IN3, IN4, ENL, ENR]:
    pi.set_mode(pin, pigpio.OUTPUT)

def set_pwm(pin, duty_percent):
    pi.set_PWM_dutycycle(pin, int(255 * duty_percent / 100))

def motor_stop():
    for p in [IN1, IN2, IN3, IN4]: pi.write(p, 0)
    set_pwm(ENL, 0); set_pwm(ENR, 0)

def motor_forward():
    pi.write(IN1, 0); pi.write(IN2, 1)
    pi.write(IN3, 0); pi.write(IN4, 1)

def motor_turn_right():
    pi.write(IN1, 1); pi.write(IN2, 0)
    pi.write(IN3, 0); pi.write(IN4, 1)

def motor_turn_left():
    pi.write(IN1, 0); pi.write(IN2, 1)
    pi.write(IN3, 1); pi.write(IN4, 0)

# ==============================================================================
# 4. ACTION ROUTINES & UI LABELS
# ==============================================================================
def action_stop():
    print(">>> BIOHAZARD/BUTTON: Stopping for 2 seconds")
    motor_stop()
    time.sleep(2)

    # Clear from the sign
    motor_forward()
    set_pwm(ENL, 55); set_pwm(ENR, 55)
    time.sleep(0.5) # Adjust this slightly if it pushes too far or too little
    motor_stop()

def action_biometric(name):
    print(f">>> {name}: Biometric displayed. Robot continues driving.")

    # Clear from the sign
    motor_forward()
    set_pwm(ENL, 55); set_pwm(ENR, 55)
    time.sleep(0.3) # Adjust this slightly if it pushes too far or too little
    motor_stop()

def action_recycle():
    print(">>> RECYCLE: Executing Dynamic 360° spin")
    pi.write(IN1, 0); pi.write(IN2, 1)
    pi.write(IN3, 1); pi.write(IN4, 0)
    set_pwm(ENL, 80); set_pwm(ENR, 80)
    
    time.sleep(0.3) 
    
    kernel = np.ones((5, 5), np.uint8)
    start_time = time.time()
    lines_crossed = 0
    currently_on_line = False
    
    while True:
        if time.time() - start_time > 3.0:
            print("    -> [FAILSAFE] 360 Spin Timeout! Aborting.")
            break
            
        rgb = picam2.capture_array()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        with shared_lock: shared["frame"] = bgr.copy()
            
        roi = bgr[141:200, 0:320]
        black_mask = cv2.inRange(roi, (0, 0, 0), (100, 100, 100))
        black_mask = cv2.erode(black_mask, kernel, iterations=2)
        cnts, _ = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        line_in_view = False
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 250:
                x, y, w, h = cv2.boundingRect(c)
                cx = int(x + (w / 2))
                if 100 < cx < 220: line_in_view = True
                    
        if line_in_view and not currently_on_line:
            lines_crossed += 1
            currently_on_line = True
            if lines_crossed >= 2:
                print("    -> 360 Spin complete.")
                break
        elif not line_in_view and currently_on_line:
            currently_on_line = False
            
        time.sleep(0.005)
    motor_stop()

# --- v8.0 UPDATED: INTERSECTION PUSH DELAY ---
def action_arrow_turn(direction):
    print(f">>> ARROW: Pushing forward for 0.1s to align wheels...")
    
    with shared_lock:
        shared["last_arrow_dir"] = direction
        
    # FORCE the robot to drive strictly straight for 100ms into the intersection
    motor_forward()
    set_pwm(ENL, 45); set_pwm(ENR, 45)
    time.sleep(0.06) 
        
    print(f">>> ARROW: Blind Turn {direction} executing!")
    if direction == "LEFT": motor_turn_left()
    elif direction == "RIGHT": motor_turn_right()
    set_pwm(ENL, 55); set_pwm(ENR, 55)
    
    time.sleep(1.0) 
    motor_stop()

SYMBOL_ACTIONS = {
    "BIOHAZARD":   action_stop,
    "BUTTON":      action_stop,
    "QR_CODE":     lambda: action_biometric("QR_CODE"),
    "FINGERPRINT": lambda: action_biometric("FINGERPRINT"),
    "RECYCLE":     action_recycle,
    "ARROW LEFT":  lambda: action_arrow_turn("LEFT"),
    "ARROW RIGHT": lambda: action_arrow_turn("RIGHT"),
}

DASHBOARD_LABELS = {
    "BIOHAZARD":   "BIOHAZARD - Stop",
    "BUTTON":      "BUTTON - Stop",
    "QR_CODE":     "QR_CODE - Biometric",
    "FINGERPRINT": "FINGERPRINT - Biometric",
    "RECYCLE":     "RECYCLE - 360 turn"
}

def execute_symbol_action(symbol):
    fn = SYMBOL_ACTIONS.get(symbol)
    if fn: fn()

# ==============================================================================
# 5. ORB TEMPLATE SETUP
# ==============================================================================
TEMPLATE_DIR  = "./orb_templates"
ORB_SYMBOLS   = ["BIOHAZARD", "RECYCLE", "QR_CODE", "FINGERPRINT", "BUTTON"]

ORB_RATIO_TEST = 0.75   
MIN_MATCH_COUNT = 6     
ORB_PARTIAL_FRAMES = 3   

CROP_SIZE = 110
CROP_X    = (320 - CROP_SIZE) // 2                                         
CROP_Y    = VISION_ROI_TOP + ((VISION_ROI_BOTTOM - VISION_ROI_TOP) - CROP_SIZE) // 2  

SYMBOL_STABLE_FRAMES = 1   
ARROW_STABLE_FRAMES  = 3 

os.makedirs(TEMPLATE_DIR, exist_ok=True)

orb = cv2.ORB_create(nfeatures=800, edgeThreshold=10, patchSize=31, fastThreshold=15)
bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
orb_templates = {}

def load_templates():
    global orb_templates
    orb_templates = {}
    for sym in ORB_SYMBOLS:
        path = os.path.join(TEMPLATE_DIR, f"{sym}.png")
        if not os.path.exists(path): continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        img_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
        kp, des = orb.detectAndCompute(img_thresh, None)
        if des is not None and len(des) > 0:
            orb_templates[sym] = (kp, des, img_thresh)
load_templates()

def save_template(symbol, bgr_frame):
    roi  = bgr_frame[CROP_Y:CROP_Y + CROP_SIZE, CROP_X:CROP_X + CROP_SIZE]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(TEMPLATE_DIR, f"{symbol}.png"), gray)
    print(f"  [ORB] Saved template: {symbol}")
    load_templates()

# ==============================================================================
# 6. ORB SYMBOL MATCHING (WITH SHAPE VALIDATOR)
# ==============================================================================
def run_orb(vision_roi_bgr, vision_roi_hsv):
    if not orb_templates: return None, 0, "ORB: no templates", None

    live_gray = cv2.cvtColor(vision_roi_bgr, cv2.COLOR_BGR2GRAY)
    live_thresh = cv2.adaptiveThreshold(live_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)   
    kp_live, des_live = orb.detectAndCompute(live_thresh, None)

    if des_live is None or len(kp_live) < 50: 
        return None, 0, "KP:<50 (Floor)", None

    best_sym = None
    max_inliers = 0
    best_H = None

    for sym, (kp_tmpl, des_tmpl, _) in orb_templates.items():
        try:
            matches = bf.knnMatch(des_tmpl, des_live, k=2)
            good = [m for m, n in matches if m.distance < ORB_RATIO_TEST * n.distance]

            if len(good) >= MIN_MATCH_COUNT:
                src_pts = np.float32([kp_tmpl[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_live[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                shape_valid = False
                
                if mask is not None and H is not None:
                    corners = np.float32([[0, 0], [0, CROP_SIZE], [CROP_SIZE, CROP_SIZE], [CROP_SIZE, 0]]).reshape(-1, 1, 2)
                    try:
                        trans_corners = cv2.perspectiveTransform(corners, H)
                        pts = np.int32(trans_corners).reshape(-1, 2)
                        area = cv2.contourArea(pts)
                        is_convex = cv2.isContourConvex(pts)
                        
                        if is_convex and 1500 < area < 35000:
                            d1 = np.linalg.norm(pts[0] - pts[1])
                            d2 = np.linalg.norm(pts[1] - pts[2])
                            d3 = np.linalg.norm(pts[2] - pts[3])
                            d4 = np.linalg.norm(pts[3] - pts[0])
                            
                            min_side = max(1.0, float(min(d1, d2, d3, d4)))
                            max_side = float(max(d1, d2, d3, d4))
                            
                            if (max_side / min_side) < 3.5:
                                shape_valid = True
                    except Exception:
                        pass
                
                if shape_valid:
                    inliers = np.sum(mask)
                    if inliers > max_inliers:
                        max_inliers = inliers
                        best_sym = sym
                        best_H = H
                else:
                    if len(good) > max_inliers:
                        max_inliers = len(good)
                        best_sym = sym
            else:
                if len(good) > max_inliers:
                    max_inliers = len(good)
                    best_sym = sym
        except Exception: pass

    confident = (max_inliers >= MIN_MATCH_COUNT) and (best_H is not None)
    flag = "FLOOR"

    if confident and best_H is not None:
        center_pt = np.float32([[[CROP_SIZE / 2.0, CROP_SIZE / 2.0]]])
        transformed_center = cv2.perspectiveTransform(center_pt, best_H)
        cx, cy = int(transformed_center[0][0][0]), int(transformed_center[0][0][1])

        h_roi, w_roi = vision_roi_hsv.shape[:2]
        cy_clamped = max(10, min(h_roi - 10, cy))
        cx_clamped = max(10, min(w_roi - 10, cx))

        center_sat = np.median(vision_roi_hsv[cy_clamped-10:cy_clamped+10, cx_clamped-10:cx_clamped+10, 1])
        center_hue = np.median(vision_roi_hsv[cy_clamped-10:cy_clamped+10, cx_clamped-10:cx_clamped+10, 0])

        if center_sat > 70:
            if 5 <= center_hue <= 40: 
                pass 
            elif 40 < center_hue <= 90:
                pass 
            else:
                confident = False           
                max_inliers = 0
                best_sym = None
                flag = "COLOR VETO"

        if confident:
            in_zone = (TRIGGER_ZONE_X1 <= cx <= TRIGGER_ZONE_X2) and (TRIGGER_ZONE_Y1 <= cy <= TRIGGER_ZONE_Y2)
            if not in_zone:
                confident = False
                flag = "OUT OF ZONE"
            else:
                flag = "TRIGGERED"
                if best_sym in ["BIOHAZARD", "BUTTON"]:
                    best_sym = "BIOHAZARD" if 10 <= center_hue <= 35 else "BUTTON"

    if not confident and flag == "FLOOR":
        flag = "FLOOR" if max_inliers < MIN_MATCH_COUNT else "SHAPE_VETO"

    partial_sym = best_sym if (max_inliers >= 3 and not confident and flag != "OUT OF ZONE") else None

    debug = f"KP:{len(kp_live)} | {best_sym}[Inliers:{max_inliers}|{flag}]"

    return (best_sym if confident else None), max_inliers, debug, partial_sym

# ==============================================================================
# 7. ARROW DETECTION
# ==============================================================================
geo_kernel = np.ones((5, 5), np.uint8)

def detect_arrows(bgr_roi, display_frame, roi_y_offset=10):
    tally       = {}
    found_boxes = []
    roi_h, roi_w = bgr_roi.shape[:2]

    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    mask_red1 = cv2.inRange(hsv, (0, 70, 70), (25, 255, 255))
    mask_red2 = cv2.inRange(hsv, (160, 70, 70), (180, 255, 255))
    mask_green = cv2.inRange(hsv, (40, 60, 60), (90, 255, 255))
    mask_blue = cv2.inRange(hsv, (100, 60, 60), (140, 255, 255))

    color_mask = cv2.bitwise_or(mask_red1, mask_red2)
    color_mask = cv2.bitwise_or(color_mask, mask_green)
    cleaned    = cv2.bitwise_or(color_mask, mask_blue)

    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  geo_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, geo_kernel)

    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 500 or area > 25000: continue

        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0 or w == 0: continue
        
        aspect_ratio = float(w) / max(1, h)
        if aspect_ratio < 0.3 or aspect_ratio > 3.5: continue
            
        if x <= 2 or y <= 2 or (x + w) >= (roi_w - 2) or (y + h) >= (roi_h - 2): continue

        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = area / float(hull_area)
        
        if solidity < 0.20 or solidity > 0.95: 
            continue

        M  = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + h // 2

        max_dist, tip_x, tip_y = 0, cx, cy
        for pt in cnt:
            px, py = pt[0][0], pt[0][1]
            d = (px - cx) ** 2 + (py - cy) ** 2
            if d > max_dist:
                max_dist = d
                tip_x, tip_y = px, py

        dx, dy    = tip_x - cx, tip_y - cy
        arrow_dir = ("Left" if dx > 0 else "Right") if abs(dx) > abs(dy) else ("Up" if dy > 0 else "Down")
        label     = f"Arrow ({arrow_dir})"

        fy = y + roi_y_offset
        
        in_zone = (TRIGGER_ZONE_X1 <= cx <= TRIGGER_ZONE_X2) and (TRIGGER_ZONE_Y1 <= cy <= TRIGGER_ZONE_Y2)
        
        if in_zone:
            found_boxes.append((x, fy, w, h, label))
            tally[label] = tally.get(label, 0) + 1
            cv2.rectangle(display_frame, (x, fy), (x + w, fy + h), (0, 255, 0), 3) # Green = TRIGGER
            cv2.putText(display_frame, f"{label} [TRIGGER]", (x, fy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        else:
            cv2.rectangle(display_frame, (x, fy), (x + w, fy + h), (0, 165, 255), 1) # Orange = WAIT
            cv2.putText(display_frame, f"{label} [WAIT]", (x, fy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

    recycle_suspected = False
    detected_dir = None

    for name in tally:
        if "Arrow (Left)" in name:
            detected_dir = "LEFT"
            break
        elif "Arrow (Right)" in name:
            detected_dir = "RIGHT"
            break

    return detected_dir, cleaned, found_boxes, recycle_suspected

# ==============================================================================
# 8. TEMPORAL SMOOTHER 
# ==============================================================================
_smoother = {"candidate": None, "count": 0, "stable": None, "type": ""}

def smooth_label(raw_label, is_symbol=False, arrow_thresh=4):
    s = _smoother
    threshold = SYMBOL_STABLE_FRAMES if is_symbol else arrow_thresh

    if raw_label == s["candidate"]: s["count"] += 1
    else:
        s["candidate"] = raw_label
        s["count"]     = 1
        s["type"]      = "symbol" if is_symbol else "arrow"

    if s["count"] >= threshold: s["stable"] = s["candidate"]
    return s["stable"]

def reset_smoother():
    _smoother["candidate"] = None
    _smoother["count"]     = 0
    _smoother["stable"]    = None
    _smoother["type"]      = ""

# ==============================================================================
# 9. THREAD 1 — LINE FOLLOWING (PID)
# ==============================================================================
PID_ROI_TOP    = 141
PID_ROI_BOTTOM = 200
FRAME_CENTER_X = 160

def get_solid_shortcut_contours(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    valid_cnts = []
    for c in cnts:
        # Just check if it's big enough to see, ignore how "solid" it is!
        if cv2.contourArea(c) > 50: 
            valid_cnts.append(c)
    return valid_cnts

def line_thread():
    Kp = 2.8
    Ki = 0.0
    Kd = 1.0

    base_speed      = 36
    max_speed       = 60
    min_speed       = 0

    previous_error = 0
    integral       = 0
    COOLDOWN_DURATION = 2.0 

    on_colour_shortcut   = False
    colour_shortcut_dir  = None
    shortcut_lost_frames = 0
    shortcut_entry_time  = 0.0   
    shortcut_lockout_until = 0.0
    
    was_lost = False
    post_corner_slowdown_until = 0.0

    print("[LINE] Thread started with Blocking Turn logic.")
    erode_kernel = np.ones((5, 5), np.uint8)   

    while True:
        with shared_lock:
            if not shared["running"]: break
        rgb = picam2.capture_array()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        with shared_lock: shared["frame"] = bgr.copy()

        now = time.time()
        with shared_lock:
            symbol      = shared["symbol"]
            in_action   = shared["symbol_action"]
            cooldown    = shared["cooldown_until"]

        if symbol is not None and not in_action and now > cooldown:
            with shared_lock:
                shared["symbol_action"]  = True
                shared["current_action"] = symbol
                shared["on_shortcut"]    = False

            execute_symbol_action(symbol)

            with shared_lock:
                shared["symbol"]         = None
                shared["symbol_action"]  = False
                shared["cooldown_until"] = time.time() + COOLDOWN_DURATION
            previous_error = 0
            integral       = 0
            shortcut_lockout_until = 0
            on_colour_shortcut = False
            continue

        if now < post_corner_slowdown_until:
            current_base_speed = 22
        else:
            current_base_speed = base_speed

        roi = bgr[PID_ROI_TOP:PID_ROI_BOTTOM, 0:320]
        pid_display = bgr.copy()
        cv2.rectangle(pid_display, (0, PID_ROI_TOP), (319, PID_ROI_BOTTOM), (200, 0, 200), 2)

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        if now > shortcut_lockout_until:
            r1 = cv2.inRange(hsv_roi, (135, 40, 40), (180, 255, 255))
            r2 = cv2.inRange(hsv_roi, (0, 40, 40), (15, 255, 255))
            red_mask = cv2.bitwise_or(r1, r2)
            red_mask = cv2.erode(red_mask, erode_kernel, iterations=2)

            yellow_mask = cv2.inRange(hsv_roi, (15, 60, 60), (55, 255, 255))
            yellow_mask = cv2.erode(yellow_mask, erode_kernel, iterations=2)
            
            cnts_red    = get_solid_shortcut_contours(red_mask)
            cnts_yellow = get_solid_shortcut_contours(yellow_mask)
        else:
            cnts_red = []
            cnts_yellow = []

        black_mask = cv2.inRange(roi, (0, 0, 0), (120, 120, 120))
        black_mask = cv2.erode(black_mask, erode_kernel, iterations=2)
        cnts_black, _  = cv2.findContours(black_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        target_c = None
        tracking_label = "LOST"
        line_color = (0, 0, 0)
        is_shortcut = False

        if len(cnts_red) > 0:
            target_c = max(cnts_red, key=cv2.contourArea)
            tracking_label = "RED (Shortcut)"
            line_color = (255, 0, 0) 
            is_shortcut = True
        elif len(cnts_yellow) > 0:
            target_c = max(cnts_yellow, key=cv2.contourArea)
            tracking_label = "YELLOW (Shortcut)"
            line_color = (0, 255, 255) 
            is_shortcut = True
        elif len(cnts_black) > 0:
            target_c = max(cnts_black, key=cv2.contourArea)
            if now < shortcut_lockout_until:
                tracking_label = f"BLACK (Lockout: {shortcut_lockout_until - now:.1f}s)"
            elif now < post_corner_slowdown_until:
                tracking_label = "BLACK (SCANNING TURN...)"
            else:
                tracking_label = "BLACK (Main)"
            line_color = (0, 0, 255) 
            is_shortcut = False

        with shared_lock: shared["on_shortcut"] = is_shortcut

        if is_shortcut:
            shortcut_lost_frames = 0
            if not on_colour_shortcut:
                with shared_lock:
                    mem_dir = shared.get("last_arrow_dir")
                    #Wipe Memory
                    shared["last_arrow_dir"] = None
                
                if mem_dir:
                    colour_shortcut_dir = mem_dir
                else:
                    c_target              = target_c
                    x_t, _, w_t, _        = cv2.boundingRect(c_target)
                    entry_error           = int(x_t + w_t / 2) - FRAME_CENTER_X
                    colour_shortcut_dir   = "RIGHT" if entry_error >= 0 else "LEFT"
                    
                on_colour_shortcut    = True
                shortcut_entry_time   = now   

            cv2.putText(pid_display, f"SHORTCUT ({tracking_label}) | Exit:{colour_shortcut_dir}",
                        (5, PID_ROI_TOP - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)

        if not is_shortcut:
            if on_colour_shortcut:
                shortcut_lost_frames += 1
                time_on_shortcut = now - shortcut_entry_time
                
                if shortcut_lost_frames > 15 and time_on_shortcut > 1.2:
                    print(f"[LINE] End of Shortcut. VISUAL Merge {colour_shortcut_dir} based on memory!")
                    if colour_shortcut_dir == "RIGHT": motor_turn_right()
                    elif colour_shortcut_dir == "LEFT": motor_turn_left()
                    
                    # Hard push, overcome friction
                    set_pwm(ENL, 80); set_pwm(ENR, 80)
                    
                    # 1. Blindly nudge for 0.5s just to get off the colored track
                    time.sleep(0.5)
                    motor_stop()

                    on_colour_shortcut      = False
                    colour_shortcut_dir     = None
                    shortcut_lost_frames    = 0
                    previous_error          = 0
                    integral                = 0
                    shortcut_lockout_until  = time.time() + 3.0
                    with shared_lock:
                        shared["on_shortcut"]    = False
                        shared["cooldown_until"] = time.time() + 0.5 
                        # Erase memory for the next run
                        #shared["last_arrow_dir"] = None 
                    continue

                # --- ANTI-JERK FIX ---
                # During the 20-frame waiting period, FREEZE the steering and coast straight.
                # Do NOT let the PID look at the black track yet!
                motor_forward()
                set_pwm(ENL, base_speed); set_pwm(ENR, base_speed)
                time.sleep(0.002)
                continue # This 'continue' skips the PID math below!

        if target_c is None:
            was_lost = True
            set_pwm(ENL, 56); set_pwm(ENR, 56)
            motor_turn_right() if previous_error > 0 else motor_turn_left()
            cv2.putText(pid_display, "LOST LINE - RECOVERING", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            with shared_lock:
                shared["pid_display"]    = pid_display
                shared["tracking_label"] = "LOST (Spinning)"
            time.sleep(0.002)
            continue
        else:
            if was_lost:
                was_lost = False
                post_corner_slowdown_until = time.time() + 1.0

        x, y, w, h = cv2.boundingRect(target_c)
        error = int(x + (w / 2)) - 160

        integral += error
        derivative = error - previous_error
        previous_error = error

        pid = Kp * error + Ki * integral + Kd * derivative
        left_speed = max(min_speed, min(max_speed, current_base_speed + pid))
        right_speed = max(min_speed, min(max_speed, current_base_speed - pid))

        set_pwm(ENL, left_speed)
        set_pwm(ENR, right_speed)
        motor_forward()

        cx = int(x + (w / 2))
        cv2.line(pid_display, (160, 215), (cx, 155), line_color, 3)
        cv2.putText(pid_display, f"err: {error}", (140, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        counts_str = f"R:{len(cnts_red)} Y:{len(cnts_yellow)} B:{len(cnts_black)}"
        cv2.putText(pid_display, counts_str, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        with shared_lock:
            shared["pid_display"]    = pid_display
            shared["tracking_label"] = tracking_label
            shared["pid_error"]      = error
        time.sleep(0.002)

    motor_stop()
    print("[LINE] Thread stopped.")

# ==============================================================================
# 10. THREAD 2 — SYMBOL & ARROW DETECTION
# ==============================================================================
def vision_thread():
    capture_mode        = False
    partial_match_label = None   
    partial_match_count = 0

    print("[VISION] Thread started.")

    while True:
        with shared_lock:
            if not shared["running"]: break
            frame_bgr  = shared["frame"]
            key_press  = shared["key_press"]
            on_shortcut = shared["on_shortcut"]   
            shared["key_press"] = None

        if frame_bgr is None:
            time.sleep(0.016)
            continue

        frame     = frame_bgr.copy()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if key_press:
            if key_press in (ord('t'), ord('T')): capture_mode = not capture_mode
            elif capture_mode and ord('1') <= key_press <= ord('9'):
                idx = key_press - ord('1')
                if idx < len(ORB_SYMBOLS): save_template(ORB_SYMBOLS[idx], frame_bgr)

        if capture_mode:
            overlay = (frame * 0.4).astype(np.uint8)
            cv2.rectangle(overlay, (CROP_X, CROP_Y), (CROP_X + CROP_SIZE, CROP_Y + CROP_SIZE), (0, 128, 255), 2)
            for i, s in enumerate(ORB_SYMBOLS):
                cv2.putText(overlay, f"{i+1}: {s}", (5, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            with shared_lock: shared["vision_display"] = overlay
            time.sleep(0.016)
            continue

        now = time.time()
        with shared_lock:
            cooldown  = shared["cooldown_until"]
            in_action = shared["symbol_action"]

        if in_action or now < cooldown:
            reset_smoother()
            partial_match_label = None
            partial_match_count = 0
            orb_debug     = "VISION PAUSED (COOLDOWN)"
            display_label = None
            with shared_lock: shared["orb_inliers"] = 0
        else:
            vision_roi = frame[VISION_ROI_TOP:VISION_ROI_BOTTOM, 0:320]
            vision_hsv = hsv_frame[VISION_ROI_TOP:VISION_ROI_BOTTOM, 0:320]
            
            orb_sym, max_inliers, orb_debug, partial_sym = run_orb(vision_roi, vision_hsv)

            with shared_lock: shared["orb_inliers"] = max_inliers

            if max_inliers >= 1:
                target_name = orb_sym if orb_sym else partial_sym
                print(f"[ORB DEBUG] Looking at: {target_name} | Matches: {max_inliers}/{MIN_MATCH_COUNT} | Details: {orb_debug}")

            if partial_sym is not None and partial_sym == partial_match_label:
                partial_match_count += 1
            else:
                partial_match_label = partial_sym
                partial_match_count = 1 if partial_sym else 0

            with shared_lock: shared["orb_partial_match"] = partial_match_label if partial_match_count >= 3 else None

            arrow_dir         = None
            recycle_suspected = False

            if not on_shortcut and orb_sym is None:
                arrow_dir, bw_mask, found_boxes, recycle_suspected = detect_arrows(vision_roi, frame, roi_y_offset=VISION_ROI_TOP)

            if recycle_suspected and orb_sym is None: orb_sym = "RECYCLE"   

            if orb_templates:
                final_label = (orb_sym if orb_sym else (f"ARROW {arrow_dir.upper()}" if arrow_dir else None))
            else:
                final_label = f"ARROW {arrow_dir.upper()}" if arrow_dir else None

            is_symbol_detection = final_label is not None and not final_label.startswith("ARROW")
            smoother_key        = ("ARROW" if (final_label and final_label.startswith("ARROW")) else final_label)
            
            smooth_key     = smooth_label(smoother_key, is_symbol=is_symbol_detection, arrow_thresh=2)
            display_label  = final_label if smooth_key == "ARROW" else smooth_key

        if display_label is not None:
            with shared_lock:
                if not shared["symbol_action"]:
                    shared["symbol"]         = display_label
                    shared["current_action"] = display_label

        vision_display = frame.copy()
        
        cv2.rectangle(vision_display, (TRIGGER_ZONE_X1, VISION_ROI_TOP + TRIGGER_ZONE_Y1), 
                                      (TRIGGER_ZONE_X2, VISION_ROI_TOP + TRIGGER_ZONE_Y2), 
                                      (0, 255, 255), 2)
        cv2.putText(vision_display, "TARGET ZONE", (TRIGGER_ZONE_X1 + 5, VISION_ROI_TOP + TRIGGER_ZONE_Y1 + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        cv2.rectangle(vision_display, (0, VISION_ROI_TOP), (319, VISION_ROI_BOTTOM), (200, 0, 200), 2)

        if on_shortcut:
            cv2.putText(vision_display, "ARROW DETECT: SUPPRESSED (ON SHORTCUT)",
                        (5, VISION_ROI_BOTTOM + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 200, 255), 1)
        if display_label:
            cv2.putText(vision_display, display_label, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        if partial_match_label:
            cv2.putText(vision_display, f"Partial[{partial_match_label}] {partial_match_count}/{ORB_PARTIAL_FRAMES}",
                        (5, VISION_ROI_BOTTOM + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 200, 255), 1)
        cv2.putText(vision_display, orb_debug, (4, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (140, 140, 140), 1)

        with shared_lock: shared["vision_display"] = vision_display
        time.sleep(0.016)

    print("[VISION] Thread stopped.")

# ==============================================================================
# 11. SCREEN 1 — Main Status Dashboard
# ==============================================================================
def build_main_display():
    with shared_lock:
        tracking    = shared.get("tracking_label",  "BLACK")
        action      = shared.get("current_action",  "IDLE")
        pid_err     = shared.get("pid_error",       0)
        cooldown    = shared.get("cooldown_until",  0)
        in_action   = shared.get("symbol_action",   False)
        frame       = shared.get("frame",           None)
        partial_sym = shared.get("orb_partial_match", None)

    now      = time.time()
    dashboard = np.zeros((240, 320, 3), dtype=np.uint8)

    cv2.rectangle(dashboard, (0, 0), (320, 30), (50, 50, 50), -1)
    cv2.putText(dashboard, "AGV MAIN STATUS v14.0",
                (50, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    track_col = (0, 255, 120) if "Shortcut" in tracking else (0, 220, 255)
    cv2.putText(dashboard, f"Track: {tracking}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, track_col, 2)

    cv2.putText(dashboard, f"PID err: {pid_err:+d}px",
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    bar_top = 74; bar_bot = 88
    clamped = max(-150, min(150, pid_err))
    bar_col = (0, 200, 80) if abs(clamped) < 30 else (0, 80, 255)
    cv2.rectangle(dashboard, (0, bar_top), (320, bar_bot), (30, 30, 30), -1)
    cv2.line(dashboard, (FRAME_CENTER_X, bar_top), (FRAME_CENTER_X, bar_bot), (80, 80, 80), 1)
    bx = FRAME_CENTER_X + clamped
    cv2.rectangle(dashboard, (min(FRAME_CENTER_X, bx), bar_top+2),
                  (max(FRAME_CENTER_X, bx), bar_bot-2), bar_col, -1)
    
    if partial_sym:
        cv2.putText(dashboard, f"Sensing: {partial_sym} (approaching...)",
                    (10, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 160, 255), 1)

    display_action = DASHBOARD_LABELS.get(action, action)

    a_col = (0, 60, 255) if in_action else (255, 255, 0)
    cv2.putText(dashboard, f"{'EXEC' if in_action else 'Last'}: {display_action}",
                (10, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.50, a_col, 2)

    cv2.putText(dashboard, "Cooldown:", (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 150, 150), 1)
    cv2.rectangle(dashboard, (10, 179), (310, 193), (35, 35, 35), -1)
    if now < cooldown:
        remain = cooldown - now
        wb     = int(300 * min(remain / 3.0, 1.0))
        cv2.rectangle(dashboard, (10, 179), (10 + wb, 193), (0, 100, 220), -1)
        cv2.putText(dashboard, f"{remain:.1f}s", (268, 191), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 200, 255), 1)

    cv2.rectangle(dashboard, (0, 208), (320, 240), (20, 20, 20), -1)
    cv2.putText(dashboard, "Q=Quit  T=Capture  1-5=SaveTemplate",
                (5, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (90, 90, 90), 1)

    camera_view = frame.copy() if frame is not None else np.zeros((240, 320, 3), dtype=np.uint8)
    return np.vstack((camera_view, dashboard))

# ==============================================================================
# 12. FLASK WEB SERVER
# ==============================================================================
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AGV Dashboard v14.0</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { background:#1e1e2e; color:#fff; font-family:monospace; text-align:center; margin:0; padding:20px; }
        h1   { color:#00ffcc; }
        .container { display:flex; flex-wrap:wrap; justify-content:center; gap:20px; margin-top:10px; }
        .stream-card { background:#282a36; border-radius:8px; padding:15px; box-shadow:0 4px 10px rgba(0,0,0,0.5); }
        img  { border:2px solid #44475a; border-radius:4px; max-width:100%; height:auto; }
        h3   { margin-top:0; color:#bd93f9; }
        .controls { margin-top:30px; background:#282a36; padding:20px; border-radius:8px; display:inline-block; }
        button { background:#6272a4; color:white; border:none; padding:12px 20px; font-size:16px; margin:5px; border-radius:4px; cursor:pointer; font-weight:bold; }
        button:hover { background:#50fa7b; color:#282a36; }
        .t-btn { background:#ffb86c; color:#282a36; }
        .fix-note { background:#383a59; border-left:4px solid #50fa7b; padding:10px; margin:10px auto; max-width:640px; text-align:left; font-size:13px; color:#fff; }
    </style>
</head>
<body>
    <h1>AGV Dashboard v14.0</h1>
    <div class="fix-note">
      <b>v14.0 The Intersection Push:</b> Added a 100ms forward push to all Arrow turns! The robot will now physically enter the center of the intersection before the blind spin activates, perfectly fixing the camera offset drop-line bug!
    </div>
    <div class="container">
        <div class="stream-card">
            <h3>1. Main Status Dashboard</h3>
            <img src="/video_main" width="320" height="480" />
        </div>
        <div class="stream-card">
            <h3>2. Debug (Vision + PID)</h3>
            <img src="/video_debug" width="320" height="240" />
        </div>
    </div>
    <div class="controls">
        <h3>Template Capture</h3>
        <button class="t-btn" onclick="sendKey('T')">Toggle Capture Box (T)</button><br><br>
        <button onclick="sendKey('1')">1: Biohazard</button>
        <button onclick="sendKey('2')">2: Recycle</button>
        <button onclick="sendKey('3')">3: QR Code</button>
        <button onclick="sendKey('4')">4: Fingerprint</button>
        <button onclick="sendKey('5')">5: Button</button>
    </div>
    <script>
        function sendKey(k) { fetch('/keypress/' + k); }
        document.addEventListener('keydown', e => {
            if (['t','T','1','2','3','4','5'].includes(e.key)) sendKey(e.key);
        });
    </script>
</body>
</html>
"""

def generate_main():
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    while True:
        with shared_lock:
            if not shared["running"]: break
        frame = build_main_display()
        ret, jpg = cv2.imencode('.jpg', frame, enc)
        if ret:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n\r\n'
        time.sleep(0.1)

def generate_debug():
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    while True:
        with shared_lock:
            if not shared["running"]: break
            pid_disp    = shared["pid_display"]
            vision_disp = shared["vision_display"]
            
        if pid_disp is not None and vision_disp is not None:
            combined = np.vstack((vision_disp[0:140, :], pid_disp[140:240, :]))
        elif vision_disp is not None:
            combined = vision_disp
        elif pid_disp is not None:
            combined = pid_disp
        else:
            combined = np.zeros((240, 320, 3), dtype=np.uint8)
            
        ret, jpg = cv2.imencode('.jpg', combined, enc)
        if ret:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n\r\n'
        time.sleep(0.1)

@app.route('/')
def index(): return render_template_string(HTML_TEMPLATE)

@app.route('/video_main')
def video_main(): return Response(generate_main(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_debug')
def video_debug(): return Response(generate_debug(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/keypress/<key>')
def keypress(key):
    with shared_lock: shared["key_press"] = ord(key)
    return "OK", 200

# ==============================================================================
# 13. LAUNCH
# ==============================================================================
if __name__ == "__main__":
    t_line   = threading.Thread(target=line_thread,   name="LINE",   daemon=True)
    t_vision = threading.Thread(target=vision_thread, name="VISION", daemon=True)
    t_line.start()
    t_vision.start()

    try:
        print("\n" + "=" * 55)
        print("  AGV v14.0 Flask Server RUNNING")
        print("  Open browser → http://<Pi-IP>:5000")
        print("=" * 55 + "\n")
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n[MAIN] Stopping...")
    finally:
        with shared_lock: shared["running"] = False
        t_line.join(timeout=2)
        t_vision.join(timeout=2)
        motor_stop()
        picam2.stop()
        pi.stop()
        print("[MAIN] Shutdown complete.")