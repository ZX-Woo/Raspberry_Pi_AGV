# ==============================================================================
# W3_Track_v4 — ORB + PhD Arrow Detection + Robust State Management
# ==============================================================================
#
# CRITICAL FIXES APPLIED:
#   - Arrow Convex Hull Solidity: Replaced basic area_ratio with mathematically 
#     robust Hull Solidity. Eliminates curvy track shadows/edges entirely.
#   - Arrow Border Rejection: Automatically deletes any shape touching the edges 
#     of the Vision ROI (since track lines touch the edges, but arrows do not).
#   - Track Eraser Dilation: The black_track_mask now slightly expands before 
#     erasing to eat the gray anti-aliased edges of the track.

import cv2
import numpy as np
import os
import time
import threading
from picamera2 import Picamera2
import pigpio
from flask import Flask, Response, render_template_string

# ==============================================================================
# 1. SHARED STATE — Thread-safe "message board" between all threads
# ==============================================================================
shared_lock = threading.Lock()

shared = {
    "frame":          None,
    "running":        True,
    "main_display":   None,  
    "pid_display":    None,  
    "vision_display": None,  

    "symbol":         None,  
    "symbol_action":  False, 
    "cooldown_until": 0.0,   

    "current_action":  "IDLE",   
    "tracking_label":  "BLACK",  
    "on_shortcut":     False,    
    "shortcut_dir":    None,     
    "shortcut_type":   None,     
    "pid_error":       0,        

    "key_press":       None,

    "arrow_exit_dir":  None,    
    "arrow_exit_armed": False,  
}

# ==============================================================================
# 2. CAMERA SETUP
# ==============================================================================
print("[INIT] Starting camera 320×240 @ 60 FPS in Native BGR888 Mode...")
picam2 = Picamera2()

camera_config = picam2.create_video_configuration(
    main={"size": (320, 240), "format": "BGR888"},
    controls={"FrameRate": 60}
)
picam2.configure(camera_config)
picam2.start()

picam2.set_controls({
    "AwbEnable":    False,
    "AeEnable":     True,
    "FrameRate":    60,
    "ColourGains":  (1.5, 1.2),
})
print("[INIT] Camera ready.")

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
    for pin in [IN1, IN2, IN3, IN4]: pi.write(pin, 0)
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
# 4. ACTION ROUTINES
# ==============================================================================
def action_stop():
    print(">>> BIOHAZARD/BUTTON: Stopping for 2 seconds")
    motor_stop()
    time.sleep(2)

def action_ignore(name):
    print(f">>> {name}: Detected and displaying. Robot is continuing to move.")
    # No motor stop, no sleep. The robot seamlessly glides over it!

def action_arrow_turn(direction):
    print(f">>> ARROW DETECTED: Forcing turn {direction}")

    if direction == "LEFT":
        pi.write(IN1, 0); pi.write(IN2, 1)
        pi.write(IN3, 1); pi.write(IN4, 0)
    elif direction == "RIGHT":
        pi.write(IN1, 1); pi.write(IN2, 0)
        pi.write(IN3, 0); pi.write(IN4, 1)
    elif direction == "DOWN":
        pi.write(IN1, 0); pi.write(IN2, 1)
        pi.write(IN3, 1); pi.write(IN4, 0)
    elif direction == "UP":
        motor_forward()

    set_pwm(ENL, 80); set_pwm(ENR, 80)
    time.sleep(0.6)
    motor_stop()

    if direction in ("LEFT", "RIGHT"):
        with shared_lock:
            shared["arrow_exit_dir"]    = direction
            shared["arrow_exit_armed"]  = True
            shared["on_shortcut"]       = True
            shared["shortcut_dir"]      = direction
            shared["shortcut_type"]     = "ARROW"
        print(f"[ACTION] Arrow shortcut armed — will force {direction} on black re-acquire.")

SYMBOL_ACTIONS = {
    "BIOHAZARD":   action_stop,
    "BUTTON":      action_stop,
    "QR_CODE":     lambda: action_ignore("QR_CODE"),
    "FINGERPRINT": lambda: action_ignore("FINGERPRINT"),
    "RECYCLE":     lambda: action_ignore("RECYCLE"),
    "ARROW LEFT":  lambda: action_arrow_turn("LEFT"),
    "ARROW RIGHT": lambda: action_arrow_turn("RIGHT"),
    "ARROW UP":    lambda: action_arrow_turn("UP"),
    "ARROW DOWN":  lambda: action_arrow_turn("DOWN"),
}

def execute_symbol_action(symbol):
    fn = SYMBOL_ACTIONS.get(symbol)
    if fn is not None:
        fn()

# ==============================================================================
# 5. ORB TEMPLATE MATCHING SETUP
# ==============================================================================
VISION_ROI_TOP    = 10    
VISION_ROI_BOTTOM = 140  

TEMPLATE_DIR   = "./orb_templates"
ORB_SYMBOLS    = ["BIOHAZARD", "RECYCLE", "QR_CODE", "FINGERPRINT", "BUTTON"]

ORB_RATIO_TEST = 0.75
ORB_MIN_FLOOR  = 5       
ORB_GAP_FACTOR = 3.0

CROP_SIZE = 110
CROP_X    = (320 - CROP_SIZE) // 2
CROP_Y    = VISION_ROI_TOP + ((VISION_ROI_BOTTOM - VISION_ROI_TOP) - CROP_SIZE) // 2 

STABLE_FRAMES_REQUIRED = 4
SWITCH_FRAMES_REQUIRED  = 4

os.makedirs(TEMPLATE_DIR, exist_ok=True)

orb = cv2.ORB_create(nfeatures=1000)
bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
orb_templates = {}

def load_templates():
    global orb_templates
    orb_templates = {}
    for sym in ORB_SYMBOLS:
        path = os.path.join(TEMPLATE_DIR, f"{sym}.png")
        if not os.path.exists(path):
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        kp, des = orb.detectAndCompute(img, None)
        if des is not None and len(des) > 0:
            orb_templates[sym] = (kp, des, img)
            print(f"  [ORB] Loaded: {sym} ({len(kp)} kp)")

load_templates()

def save_template(symbol, bgr_frame):
    roi  = bgr_frame[CROP_Y:CROP_Y + CROP_SIZE, CROP_X:CROP_X + CROP_SIZE]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(TEMPLATE_DIR, f"{symbol}.png"), gray)
    print(f"  [ORB] Saved template: {symbol}")
    load_templates() 

# ==============================================================================
# 6. VISION MATH — ORB Symbol Matching
# ==============================================================================
def run_orb(bgr_frame, hsv_frame):
    if not orb_templates:
        return None, False, "ORB: no templates loaded"

    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    kp_live, des_live = orb.detectAndCompute(gray, None)

    if des_live is None or len(kp_live) == 0:
        return None, False, "KP:0 (blind)"

    results = []
    for sym, (kp_tmpl, des_tmpl, _) in orb_templates.items():
        try:
            matches = bf.knnMatch(des_tmpl, des_live, k=2)
            good    = [
                m for pair in matches if len(pair) == 2
                for m, n in [pair]
                if m.distance < ORB_RATIO_TEST * n.distance 
            ]
            results.append((sym, len(good)))
        except Exception:
            results.append((sym, 0))

    results.sort(key=lambda x: x[1], reverse=True)

    best_sym     = results[0][0] if results else None
    best_count   = results[0][1] if results else 0
    second_count = results[1][1] if len(results) >= 2 else 0

    above_floor = best_count   >= ORB_MIN_FLOOR
    clear_gap   = (len(results) < 2) or (best_count >= ORB_GAP_FACTOR * second_count)
    confident   = above_floor and clear_gap

    if confident and best_sym in ["BIOHAZARD", "BUTTON"]:
        h, w       = bgr_frame.shape[:2]
        center_hue = np.median(hsv_frame[h//2-10:h//2+10, w//2-10:w//2+10, 0])
        best_sym = "BIOHAZARD" if 10 <= center_hue <= 35 else "BUTTON"

    flag = ("OK" if confident else
            (f"FLOOR<{ORB_MIN_FLOOR}" if not above_floor else "GAP"))

    debug = f"KP:{len(kp_live)} | {best_sym}[{best_count}m|{flag}]"
    return (best_sym if confident else None), True, debug

# ==============================================================================
# 7. VISION MATH — Arrow Detection 
# ==============================================================================
geo_kernel = np.ones((5, 5), np.uint8)

def detect_arrows(bgr_roi, display_frame, roi_y_offset=10):
    tally = {}
    found_boxes = []

    # Get dimensions for Border Rejection
    roi_h, roi_w = bgr_roi.shape[:2]

    yuv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2YUV)
    y_channel = yuv[:, :, 0]
    blurred = cv2.GaussianBlur(y_channel, (7, 7), 0)

    # Otsu handles shadows, but turns track edges white
    _, cleaned = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate the track eraser to eat the anti-aliased gray edges!
    black_track_mask = cv2.inRange(bgr_roi, (0, 0, 0), (100, 100, 100))
    fat_track_mask = cv2.dilate(black_track_mask, geo_kernel, iterations=1)
    cleaned = cv2.bitwise_and(cleaned, cv2.bitwise_not(fat_track_mask))

    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  geo_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, geo_kernel)

    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 800 or area > 12000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0 or w == 0 or w > 200 or h > 200:
            continue

        # Border Rejection
        # Track lines enter/exit the edges of the frame. Arrows are stickers in the middle.
        if x <= 5 or y <= 5 or (x + w) >= (roi_w - 5) or (y + h) >= (roi_h - 5):
            continue

        # Convex Hull Solidity
        # Curvy track shadows have extremely low solidity (< 0.30).
        # Arrows are solid geometric bodies (0.40 to 0.80).
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: 
            continue
            
        solidity = area / float(hull_area)
        if solidity < 0.40 or solidity > 0.80:
            continue

        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + h // 2

        max_dist = 0
        tip_x, tip_y = cx, cy
        for pt in cnt:
            px, py = pt[0][0], pt[0][1]
            dist = (px - cx)**2 + (py - cy)**2 
            if dist > max_dist:
                max_dist = dist
                tip_x, tip_y = px, py

        dx, dy = tip_x - cx, tip_y - cy
        
        arrow_dir = ("Left" if dx > 0 else "Right") if abs(dx) > abs(dy) else ("Up" if dy > 0 else "Down")
        label = f"Arrow ({arrow_dir})"

        fy = y + roi_y_offset
        found_boxes.append((x, fy, w, h, label))
        
        # Drawn in distinct blue-purple to verify the new geometric logic is firing!
        cv2.rectangle(display_frame, (x, fy), (x + w, fy + h), (255, 100, 50), 2)
        cv2.putText(display_frame, label, (x, fy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 50), 2)
        tally[label] = tally.get(label, 0) + 1

    detected_dir = None
    for name in tally:
        if "Arrow" in name:
            detected_dir = name.split("(")[1].strip(")")
            break

    return detected_dir, cleaned, found_boxes

# ==============================================================================
# 8. TEMPORAL SMOOTHER 
# ==============================================================================
_smoother = {"candidate": None, "count": 0, "stable": None}

def smooth_label(raw_label):
    s = _smoother
    if raw_label == s["candidate"]:
        s["count"] += 1
    else:
        s["candidate"] = raw_label
        s["count"]     = 1

    threshold = SWITCH_FRAMES_REQUIRED if s["stable"] is not None else STABLE_FRAMES_REQUIRED
    if s["count"] >= threshold:
        s["stable"] = s["candidate"]

    return s["stable"]

def reset_smoother():
    _smoother["candidate"] = None
    _smoother["count"]     = 0
    _smoother["stable"]    = None

# ==============================================================================
# 9. THREAD 1 — LINE FOLLOWING (PID)
# ==============================================================================
PID_ROI_TOP    = 141    
PID_ROI_BOTTOM = 200
PID_ROI_MID    = (PID_ROI_TOP + PID_ROI_BOTTOM) // 2   
FRAME_CENTER_X = 160   

def line_thread():
    Kp = 0.5    
    Ki = 0.0    
    Kd = 1.0    

    base_speed = 35    
    max_speed  = 70    
    min_speed  = 0     

    previous_error = 0
    integral       = 0
    COOLDOWN_DURATION = 2.0  

    on_colour_shortcut   = False      
    colour_shortcut_dir  = None       
    shortcut_lost_frames = 0          

    print("[LINE] Thread started.")

    while True:
        with shared_lock:
            if not shared["running"]:
                break

        rgb = picam2.capture_array()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) 

        with shared_lock:
            shared["frame"] = bgr.copy()

        now = time.time()
        with shared_lock:
            symbol    = shared["symbol"]
            in_action = shared["symbol_action"]
            cooldown  = shared["cooldown_until"]

        if symbol is not None and not in_action and now > cooldown:
            with shared_lock:
                shared["symbol_action"]  = True
                shared["current_action"] = symbol
            
            on_colour_shortcut   = False
            colour_shortcut_dir  = None
            shortcut_lost_frames = 0
            with shared_lock:
                shared["on_shortcut"]      = False
                shared["shortcut_dir"]     = None
                shared["shortcut_type"]    = None
                shared["arrow_exit_armed"] = False
                shared["arrow_exit_dir"]   = None

            execute_symbol_action(symbol)  
            
            with shared_lock:
                shared["symbol"]         = None
                shared["symbol_action"]  = False
                shared["cooldown_until"] = time.time() + COOLDOWN_DURATION
                shared["current_action"] = "IDLE"
            
            previous_error = 0
            integral       = 0
            continue

        with shared_lock:
            arrow_exit_armed = shared["arrow_exit_armed"]
            arrow_exit_dir   = shared["arrow_exit_dir"]

        roi = bgr[PID_ROI_TOP:PID_ROI_BOTTOM, 0:320] 

        pid_display = bgr.copy()

        cv2.rectangle(pid_display,
                      (0, PID_ROI_TOP), (319, PID_ROI_BOTTOM),
                      (200, 0, 200), 2)

        cv2.line(pid_display,
                 (0, PID_ROI_MID), (319, PID_ROI_MID),
                 (200, 0, 200), 1)

        main_mask = cv2.inRange(roi, (0, 0, 0), (100, 100, 100))

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        m1 = cv2.inRange(hsv_roi, (0,   100, 100), (10,  255, 255))
        m2 = cv2.inRange(hsv_roi, (160, 100, 100), (180, 255, 255))
        red_mask = cv2.bitwise_or(m1, m2)
        yellow_mask = cv2.inRange(hsv_roi, (15, 100, 100), (35, 255, 255))

        clean_red    = cv2.erode(red_mask, np.ones((5, 5), np.uint8), iterations=2)
        clean_yellow = cv2.erode(yellow_mask, np.ones((5, 5), np.uint8), iterations=2)

        cnts_red, _    = cv2.findContours(clean_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts_yellow, _ = cv2.findContours(clean_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        active_mask    = None
        detected_color = None
        contours_target = []
        tracking_label = "BLACK"
        contours       = []

        if len(cnts_red) > 0 and len(cnts_yellow) > 0:
            max_r = cv2.contourArea(max(cnts_red, key=cv2.contourArea))
            max_y = cv2.contourArea(max(cnts_yellow, key=cv2.contourArea))
            if max_r > max_y:
                contours_target = cnts_red
                detected_color = "RED"
                active_mask = clean_red
            else:
                contours_target = cnts_yellow
                detected_color = "YELLOW"
                active_mask = clean_yellow
        elif len(cnts_red) > 0:
            contours_target = cnts_red
            detected_color = "RED"
            active_mask = clean_red
        elif len(cnts_yellow) > 0:
            contours_target = cnts_yellow
            detected_color = "YELLOW"
            active_mask = clean_yellow

        if active_mask is not None:
            tracking_label = f"{detected_color} (Shortcut)"
            contours       = contours_target
            shortcut_lost_frames = 0

            if not on_colour_shortcut:
                c_target         = max(contours_target, key=cv2.contourArea)
                x_t, _, w_t, _   = cv2.boundingRect(c_target)
                entry_error      = int(x_t + w_t / 2) - FRAME_CENTER_X
                colour_shortcut_dir = "RIGHT" if entry_error >= 0 else "LEFT"
                on_colour_shortcut  = True
                with shared_lock:
                    shared["on_shortcut"]   = True
                    shared["shortcut_dir"]  = colour_shortcut_dir
                    shared["shortcut_type"] = "COLOUR"

            cv2.putText(pid_display,
                        f"SHORTCUT ACTIVE ({detected_color}) | Exit: {colour_shortcut_dir}",
                        (5, PID_ROI_TOP - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)

        if active_mask is None:
            active_mask    = cv2.erode(main_mask, np.ones((5, 5), np.uint8), iterations=2)
            contours, _    = cv2.findContours(active_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            tracking_label = "BLACK (Main Route)"

            if on_colour_shortcut:
                shortcut_lost_frames += 1
                if shortcut_lost_frames > 2:  
                    if colour_shortcut_dir == "RIGHT":
                        motor_turn_right()
                    elif colour_shortcut_dir == "LEFT":
                        motor_turn_left()
                    set_pwm(ENL, 80); set_pwm(ENR, 80)
                    time.sleep(0.5)
                    motor_stop()

                    on_colour_shortcut   = False
                    colour_shortcut_dir  = None
                    shortcut_lost_frames = 0
                    previous_error       = 0
                    integral             = 0
                    
                    with shared_lock:
                        shared["on_shortcut"]   = False
                        shared["shortcut_dir"]  = None
                        shared["shortcut_type"] = None
                        shared["cooldown_until"] = time.time() + 1.0 
                    continue

            elif arrow_exit_armed:
                if arrow_exit_dir == "RIGHT":
                    motor_turn_right()
                elif arrow_exit_dir == "LEFT":
                    motor_turn_left()
                set_pwm(ENL, 80); set_pwm(ENR, 80)
                time.sleep(0.5)
                motor_stop()

                with shared_lock:
                    shared["arrow_exit_armed"] = False
                    shared["arrow_exit_dir"]   = None
                    shared["on_shortcut"]      = False
                    shared["shortcut_dir"]     = None
                    shared["shortcut_type"]    = None
                    shared["cooldown_until"] = time.time() + 1.0
                previous_error = 0
                integral       = 0
                continue

        if len(contours) == 0:
            set_pwm(ENL, 100); set_pwm(ENR, 100)
            if previous_error > 0:
                motor_turn_right()  
            else:
                motor_turn_left()   

            with shared_lock:
                shared["pid_display"]    = pid_display
                shared["tracking_label"] = tracking_label
            time.sleep(0.002)
            continue

        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        line_cx = int(x + w / 2)      
        error   = line_cx - FRAME_CENTER_X    

        integral   += error                   
        derivative  = error - previous_error  
        previous_error = error

        pid = Kp * error + Ki * integral + Kd * derivative

        left_speed  = max(min_speed, min(max_speed, base_speed + pid))
        right_speed = max(min_speed, min(max_speed, base_speed - pid))

        set_pwm(ENL, left_speed)
        set_pwm(ENR, right_speed)
        motor_forward()

        roi_line_y = PID_ROI_TOP + (y + h // 2)  
        cv2.line(pid_display,
                 (FRAME_CENTER_X, 230),  
                 (line_cx, roi_line_y),  
                 (255, 0, 0), 3)

        cv2.circle(pid_display, (line_cx, roi_line_y), 6, (0, 0, 255), -1)

        mem_txt = f" | Mem:{shared.get('shortcut_dir', '')}" if shared.get("on_shortcut") else ""
        cv2.putText(pid_display, f"Track: {tracking_label}{mem_txt}",
                    (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(pid_display,
                    f"err:{error:+d}  pid:{pid:.1f}  L:{left_speed:.0f} R:{right_speed:.0f}",
                    (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 180, 0), 1)

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
VISION_ROI_MID    = (VISION_ROI_TOP + VISION_ROI_BOTTOM) // 2

ACTION_LABELS = {
    "ARROW LEFT":  "→ TURN LEFT",
    "ARROW RIGHT": "→ TURN RIGHT",
    "ARROW UP":    "→ GO FORWARD",
    "ARROW DOWN":  "→ U-TURN",
    "BIOHAZARD":   "→ STOP 2s",
    "BUTTON":      "→ STOP 2s",
    "QR_CODE":     "→ DETECTED",
    "FINGERPRINT": "→ DETECTED",
    "RECYCLE":     "→ DETECTED",
}

def vision_thread():
    capture_mode   = False   
    orb_skip_count = 0       

    print("[VISION] Thread started.")

    while True:
        with shared_lock:
            if not shared["running"]: break
            frame_bgr = shared["frame"]    
            key_press = shared["key_press"]
            shared["key_press"] = None      

        if frame_bgr is None:
            time.sleep(0.016)
            continue

        frame     = frame_bgr.copy()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

        if key_press:
            if key_press in (ord('t'), ord('T')):
                capture_mode = not capture_mode
            elif capture_mode and ord('1') <= key_press <= ord('9'):
                idx = key_press - ord('1')
                if idx < len(ORB_SYMBOLS):
                    save_template(ORB_SYMBOLS[idx], frame_bgr)

        if capture_mode:
            overlay = (frame * 0.4).astype(np.uint8)  
            cv2.rectangle(overlay,
                          (CROP_X, CROP_Y), (CROP_X + CROP_SIZE, CROP_Y + CROP_SIZE),
                          (0, 128, 255), 2)
            for i, s in enumerate(ORB_SYMBOLS):
                cv2.putText(overlay, f"{i+1}: {s}", (5, 20 + i * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            with shared_lock:
                shared["vision_display"] = overlay
            time.sleep(0.016)
            continue

        now = time.time()
        with shared_lock:
            cooldown  = shared["cooldown_until"]
            in_action = shared["symbol_action"]

        if in_action or now < cooldown:
            reset_smoother()
            orb_debug     = "VISION PAUSED (COOLDOWN)"
            display_label = None
        else:
            vision_roi = frame[VISION_ROI_TOP:VISION_ROI_BOTTOM, 0:320] 
            
            orb_sym   = None
            orb_debug = "ORB: skip"
            orb_skip_count += 1
            if orb_skip_count >= 3:
                orb_skip_count = 0
                orb_sym, _, orb_debug = run_orb(frame, hsv_frame)

            if orb_sym:
                arrow_dir = None
            else:
                arrow_dir, bw_mask, found_boxes = detect_arrows(vision_roi, frame, roi_y_offset=VISION_ROI_TOP)

            if orb_templates:
                final_label = (orb_sym if orb_sym
                               else (f"ARROW {arrow_dir.upper()}" if arrow_dir else None))
            else:
                final_label = f"ARROW {arrow_dir.upper()}" if arrow_dir else None

            smoother_key  = ("ARROW" if (final_label and final_label.startswith("ARROW"))
                              else final_label)
            smooth_key    = smooth_label(smoother_key)
            display_label = final_label if smooth_key == "ARROW" else smooth_key

        if display_label is not None:
            with shared_lock:
                if not shared["symbol_action"]:
                    shared["symbol"]        = display_label
                    shared["current_action"] = display_label

        vision_display = frame.copy()

        cv2.rectangle(vision_display,
                      (0, VISION_ROI_TOP), (319, VISION_ROI_BOTTOM),
                      (200, 0, 200), 2)
        cv2.line(vision_display,
                 (0, VISION_ROI_MID), (319, VISION_ROI_MID),
                 (200, 0, 200), 1)

        if display_label:
            action_txt = ACTION_LABELS.get(display_label, "")
            cv2.putText(vision_display, display_label,
                        (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(vision_display, action_txt,
                        (5, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 100), 1)

        cv2.putText(vision_display, orb_debug,
                    (4, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (140, 140, 140), 1)

        with shared_lock:
            shared["vision_display"] = vision_display

        time.sleep(0.016) 

    print("[VISION] Thread stopped.")

# ==============================================================================
# 11. SCREEN BUILDER — Main Dashboard
# ==============================================================================
def build_main_display():
    with shared_lock:
        tracking  = shared.get("tracking_label",  "BLACK")
        action    = shared.get("current_action",  "IDLE")
        on_sc     = shared.get("on_shortcut",     False)
        sc_dir    = shared.get("shortcut_dir",    None)
        sc_type   = shared.get("shortcut_type",   None)
        pid_err   = shared.get("pid_error",       0)
        cooldown  = shared.get("cooldown_until",  0)
        in_action = shared.get("symbol_action",   False)
        frame     = shared.get("frame", None)

    now = time.time()
    dashboard = np.zeros((240, 320, 3), dtype=np.uint8)

    cv2.rectangle(dashboard, (0, 0), (320, 30), (50, 50, 50), -1)
    cv2.putText(dashboard, "AGV MAIN STATUS",
                (60, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    track_col = (0, 255, 120) if "Shortcut" in tracking else (0, 220, 255)
    cv2.putText(dashboard, f"Track: {tracking}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_col, 2)

    bar_top  = 68
    bar_bot  = 86
    clamped  = max(-150, min(150, pid_err))  
    bar_col  = (0, 200, 80) if abs(clamped) < 30 else (0, 80, 255)  
    cv2.rectangle(dashboard, (0, bar_top), (320, bar_bot), (30, 30, 30), -1)
    cv2.line(dashboard, (FRAME_CENTER_X, bar_top), (FRAME_CENTER_X, bar_bot), (80, 80, 80), 1)
    bar_x = FRAME_CENTER_X + clamped
    cv2.rectangle(dashboard,
                  (min(FRAME_CENTER_X, bar_x), bar_top + 2),
                  (max(FRAME_CENTER_X, bar_x), bar_bot - 2),
                  bar_col, -1)
    cv2.putText(dashboard, f"PID err: {pid_err:+d} px",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    if on_sc:
        sc_label = f"SHORTCUT [{sc_type}]  Entry-Exit: {sc_dir}"
        cv2.putText(dashboard, sc_label,
                    (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 2)
    else:
        cv2.putText(dashboard, "Shortcut: none",
                    (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 70, 70), 1)

    a_col = (0, 60, 255) if in_action else (255, 255, 0)
    prefix = "EXECUTING: " if in_action else "Last:  "
    cv2.putText(dashboard, f"{prefix}{action}",
                (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.52, a_col, 2)

    cv2.putText(dashboard, "Cooldown:", (10, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1)
    cv2.rectangle(dashboard, (10, 182), (310, 198), (35, 35, 35), -1)
    if now < cooldown:
        remain = cooldown - now
        w_bar  = int(300 * min(remain / 3.0, 1.0))
        cv2.rectangle(dashboard, (10, 182), (10 + w_bar, 198), (0, 100, 220), -1)
        cv2.putText(dashboard, f"{remain:.1f}s", (270, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 255), 1)

    if frame is None:
        camera_view = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(camera_view, "WAITING FOR CAMERA...", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    else:
        camera_view = frame.copy()

    stacked_canvas = np.vstack((camera_view, dashboard))
    return stacked_canvas

# ==============================================================================
# 12. FLASK WEB SERVER & APP LOGIC
# ==============================================================================
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AGV Web Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { background-color: #1e1e2e; color: #fff; font-family: monospace; text-align: center; margin: 0; padding: 20px; }
        h1 { color: #00ffcc; }
        .container { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin-top: 10px; }
        .stream-card { background: #282a36; border-radius: 8px; padding: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }
        img { border: 2px solid #44475a; border-radius: 4px; max-width: 100%; height: auto; }
        h3 { margin-top: 0; color: #bd93f9; }
        .controls { margin-top: 30px; background: #282a36; padding: 20px; border-radius: 8px; display: inline-block; }
        button { background: #6272a4; color: white; border: none; padding: 12px 20px; font-size: 16px; margin: 5px; border-radius: 4px; cursor: pointer; font-weight: bold;}
        button:hover { background: #50fa7b; color: #282a36; }
        .t-btn { background: #ffb86c; color: #282a36; }
    </style>
</head>
<body>
    <h1>AGV Headless Dashboard</h1>
    
    <div class="container">
        <div class="stream-card">
            <h3>1. Main Status Dashboard</h3>
            <img src="/video_main" width="320" height="480" />
        </div>
        <div class="stream-card">
            <h3>2. Combined Debug (Vision + PID)</h3>
            <img src="/video_debug" width="320" height="240" />
        </div>
    </div>

    <div class="controls">
        <h3>Template Capture Controls</h3>
        <p><i>You can also just press the keys on your keyboard!</i></p>
        <button class="t-btn" onclick="sendKey('T')">Toggle Capture Box (T)</button><br><br>
        <button onclick="sendKey('1')">Save 1 (Biohazard)</button>
        <button onclick="sendKey('2')">Save 2 (Recycle)</button>
        <button onclick="sendKey('3')">Save 3 (QR Code)</button>
        <button onclick="sendKey('4')">Save 4 (Fingerprint)</button>
        <button onclick="sendKey('5')">Save 5 (Button)</button>
    </div>

    <script>
        function sendKey(key) {
            fetch('/keypress/' + key);
        }
        // Listen for actual keyboard presses on the webpage
        document.addEventListener('keydown', function(event) {
            const validKeys = ['t', 'T', '1', '2', '3', '4', '5'];
            if (validKeys.includes(event.key)) {
                sendKey(event.key);
            }
        });
    </script>
</body>
</html>
"""

def generate_main():
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50] 
    
    while True:
        with shared_lock:
            if not shared["running"]: break
        
        main_disp = build_main_display()
        ret, jpeg = cv2.imencode('.jpg', main_disp, encode_param)
        
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.1) 

def generate_debug():
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    
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

        ret, jpeg = cv2.imencode('.jpg', combined, encode_param)
        
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.1) 

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_main')
def video_main():
    return Response(generate_main(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_debug')
def video_debug():
    return Response(generate_debug(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/keypress/<key>')
def keypress(key):
    with shared_lock:
        shared["key_press"] = ord(key)
    return "OK", 200

# ==============================================================================
# 13. LAUNCH SCRIPT
# ==============================================================================
if __name__ == "__main__":
    t_line   = threading.Thread(target=line_thread,   name="LINE",   daemon=True)
    t_vision = threading.Thread(target=vision_thread, name="VISION", daemon=True)

    t_line.start()
    t_vision.start()

    try:
        print("\n" + "="*50)
        print(" [MAIN] AGV Flask Server is RUNNING!")
        print(" -> Open a browser on your laptop/phone.")
        print(f" -> Go to: http://localhost:5000  (or your Pi's IP address)")
        print("="*50 + "\n")
        
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False, use_reloader=False)

    except KeyboardInterrupt:
        print("\n[MAIN] Keyboard interrupt received.")
        
    finally:
        with shared_lock: 
            shared["running"] = False
            
        print("[MAIN] Shutting down threads...")
        t_line.join(timeout=2)
        t_vision.join(timeout=2)
        
        print("[MAIN] Stopping hardware...")
        motor_stop()              
        picam2.stop()            
        pi.stop()                
        print("[MAIN] Shutdown complete.")