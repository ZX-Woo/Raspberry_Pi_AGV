# ==============================================================================
# AGV COMBINED SYSTEM v5 — ORB + HOMOGRAPHY + ARROW DETECTION + FLASK UI
# ==============================================================================
#
# FULL FEATURE LIST:
#   1. Tight 110x110 ORB Crop: Takes a perfect "mugshot" of the symbol without 
#      memorizing background floor noise.
#   2. Planar Homography: Searches the ENTIRE 320x130 purple box to find the 
#      symbol, even if it's off-center or skewed during a sharp turn.
#   3. Headless Web UI: Runs on Flask. Access via http://<YOUR_PI_IP>:5000.
#   4. Arrow Solidity Filter: Eliminates track shadows and curvy lines.
#   5. 3-Second Shortcut Lockout: Physically blinds the robot to red/yellow for 
#      3 seconds after an exit turn to prevent infinite loops!
#   6. Non-Stop Actions: QR, Fingerprint, and Recycle display on the UI but 
#      allow the robot to keep driving smoothly.
#
# ==============================================================================

import cv2
import numpy as np
import os
import time
import threading
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
    "shortcut_dir":    None,
    "shortcut_type":   None,
    "pid_error":       0,

    "key_press":       None,

    "arrow_exit_dir":   None,
    "arrow_exit_armed": False,

    "orb_partial_match": None,   
}

# ==============================================================================
# 2. CAMERA SETUP
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

def action_biometric(name):
    print(f">>> {name}: Biometric displayed. Robot continues driving.")

def action_recycle():
    print(">>> RECYCLE: 360 turn displayed. Robot continues driving.")

def action_arrow_turn(direction):
    print(f">>> ARROW: Forcing turn {direction}")

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
        print(f"[ACTION] Arrow exit armed — will force {direction} after 8 black frames.")

SYMBOL_ACTIONS = {
    "BIOHAZARD":   action_stop,
    "BUTTON":      action_stop,
    "QR_CODE":     lambda: action_biometric("QR_CODE"),
    "FINGERPRINT": lambda: action_biometric("FINGERPRINT"),
    "RECYCLE":     action_recycle,
    "ARROW LEFT":  lambda: action_arrow_turn("LEFT"),
    "ARROW RIGHT": lambda: action_arrow_turn("RIGHT"),
    "ARROW UP":    lambda: action_arrow_turn("UP"),
    "ARROW DOWN":  lambda: action_arrow_turn("DOWN"),
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
    if fn:
        fn()

# ==============================================================================
# 5. ORB TEMPLATE SETUP
# ==============================================================================
VISION_ROI_TOP    = 10
VISION_ROI_BOTTOM = 140
VISION_ROI_MID    = (VISION_ROI_TOP + VISION_ROI_BOTTOM) // 2

TEMPLATE_DIR  = "./orb_templates"
ORB_SYMBOLS   = ["BIOHAZARD", "RECYCLE", "QR_CODE", "FINGERPRINT", "BUTTON"]

ORB_RATIO_TEST = 0.75   
MIN_MATCH_COUNT = 5      
ORB_PARTIAL_FRAMES = 5   

# The "Mugshot" Box
CROP_SIZE = 110
CROP_X    = (320 - CROP_SIZE) // 2                                         
CROP_Y    = VISION_ROI_TOP + ((VISION_ROI_BOTTOM - VISION_ROI_TOP) - CROP_SIZE) // 2  

SYMBOL_STABLE_FRAMES = 2   
ARROW_STABLE_FRAMES  = 4   

os.makedirs(TEMPLATE_DIR, exist_ok=True)

orb = cv2.ORB_create(nfeatures=800)
bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
orb_templates = {}

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

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
        img_eq = clahe.apply(img)
        kp, des = orb.detectAndCompute(img_eq, None)
        if des is not None and len(des) > 0:
            orb_templates[sym] = (kp, des, img_eq)
            print(f"  [ORB] Loaded: {sym} ({len(kp)} kp)")

load_templates()

def save_template(symbol, bgr_frame):
    roi  = bgr_frame[CROP_Y:CROP_Y + CROP_SIZE, CROP_X:CROP_X + CROP_SIZE]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(TEMPLATE_DIR, f"{symbol}.png"), gray)
    print(f"  [ORB] Saved template: {symbol}")
    load_templates()

# ==============================================================================
# 6. ORB SYMBOL MATCHING 
# ==============================================================================
def run_orb(vision_roi_bgr, vision_roi_hsv):
    if not orb_templates:
        return None, False, "ORB: no templates", None

    live_gray = cv2.cvtColor(vision_roi_bgr, cv2.COLOR_BGR2GRAY)
    live_eq   = clahe.apply(live_gray)   

    kp_live, des_live = orb.detectAndCompute(live_eq, None)

    if des_live is None or len(kp_live) < 5:
        return None, False, "KP:<5 (blind)", None

    best_sym = None
    max_inliers = 0
    best_H = None

    for sym, (kp_tmpl, des_tmpl, _) in orb_templates.items():
        try:
            matches = bf.knnMatch(des_tmpl, des_live, k=2)
            good = []
            for m, n in matches:
                if m.distance < ORB_RATIO_TEST * n.distance:
                    good.append(m)

            if len(good) >= MIN_MATCH_COUNT:
                src_pts = np.float32([kp_tmpl[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_live[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if mask is not None:
                    inliers = np.sum(mask)
                    if inliers > max_inliers:
                        max_inliers = inliers
                        best_sym = sym
                        best_H = H
        except Exception:
            pass

    confident = max_inliers >= MIN_MATCH_COUNT  
    partial_sym = best_sym if (max_inliers >= 3 and not confident) else None

    if confident and best_sym in ["BIOHAZARD", "BUTTON"] and best_H is not None:
        center_pt = np.float32([[[CROP_SIZE / 2.0, CROP_SIZE / 2.0]]])
        transformed_center = cv2.perspectiveTransform(center_pt, best_H)
        cx, cy = int(transformed_center[0][0][0]), int(transformed_center[0][0][1])

        h_roi, w_roi = vision_roi_hsv.shape[:2]
        cy = max(10, min(h_roi - 10, cy))
        cx = max(10, min(w_roi - 10, cx))

        center_hue = np.median(vision_roi_hsv[cy-10:cy+10, cx-10:cx+10, 0])
        best_sym   = "BIOHAZARD" if 10 <= center_hue <= 35 else "BUTTON"

    flag  = "OK" if confident else ("FLOOR" if max_inliers < MIN_MATCH_COUNT else "GAP")
    debug = f"KP:{len(kp_live)} | {best_sym}[Inliers:{max_inliers}|{flag}]"

    return (best_sym if confident else None), (max_inliers > 0), debug, partial_sym

# ==============================================================================
# 7. ARROW DETECTION with RECYCLE VETO
# ==============================================================================
geo_kernel = np.ones((5, 5), np.uint8)

def detect_arrows(bgr_roi, display_frame, roi_y_offset=10):
    tally       = {}
    found_boxes = []
    centroids   = []   

    roi_h, roi_w = bgr_roi.shape[:2]

    yuv       = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2YUV)
    y_channel = yuv[:, :, 0]
    blurred   = cv2.GaussianBlur(y_channel, (7, 7), 0)

    _, cleaned = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    black_track_mask = cv2.inRange(bgr_roi, (0, 0, 0), (100, 100, 100))
    fat_track_mask   = cv2.dilate(black_track_mask, geo_kernel, iterations=1)
    cleaned          = cv2.bitwise_and(cleaned, cv2.bitwise_not(fat_track_mask))

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

        if x <= 5 or y <= 5 or (x + w) >= (roi_w - 5) or (y + h) >= (roi_h - 5):
            continue

        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / float(hull_area)
        if solidity < 0.40 or solidity > 0.85:
            continue

        M  = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x + w // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y + h // 2

        max_dist, tip_x, tip_y = 0, cx, cy
        for pt in cnt:
            px, py = pt[0][0], pt[0][1]
            d = (px - cx) ** 2 + (py - cy) ** 2
            if d > max_dist:
                max_dist = d    # Safe!
                tip_x, tip_y = px, py

        dx, dy    = tip_x - cx, tip_y - cy
        arrow_dir = ("Left" if dx > 0 else "Right") if abs(dx) > abs(dy) else ("Up" if dy > 0 else "Down")
        label     = f"Arrow ({arrow_dir})"

        fy = y + roi_y_offset
        found_boxes.append((x, fy, w, h, label))
        centroids.append((cx, cy))

        cv2.rectangle(display_frame, (x, fy), (x + w, fy + h), (255, 100, 50), 2)
        cv2.putText(display_frame, label, (x, fy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 50), 2)
        tally[label] = tally.get(label, 0) + 1

    recycle_suspected = False
    if len(centroids) >= 3:
        xs = [c[0] for c in centroids]
        ys = [c[1] for c in centroids]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)

        if span_x < 160 and span_y < 160:
            pts = centroids[:3]
            d01 = ((pts[0][0]-pts[1][0])**2 + (pts[0][1]-pts[1][1])**2) ** 0.5
            d12 = ((pts[1][0]-pts[2][0])**2 + (pts[1][1]-pts[2][1])**2) ** 0.5
            d02 = ((pts[0][0]-pts[2][0])**2 + (pts[0][1]-pts[2][1])**2) ** 0.5
            sides    = sorted([d01, d12, d02])
            if sides[0] > 0 and sides[2] / sides[0] < 2.5:
                recycle_suspected = True
                for (bx, bfy, bw, bh, _) in found_boxes:
                    cv2.rectangle(display_frame, (bx, bfy), (bx+bw, bfy+bh), (0, 255, 255), 2)
                cv2.putText(display_frame, "RECYCLE? (veto arrows)",
                            (5, roi_y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                tally = {}   

    detected_dir = None
    if not recycle_suspected:
        for name in tally:
            if "Arrow" in name:
                detected_dir = name.split("(")[1].strip(")")
                break

    return detected_dir, cleaned, found_boxes, recycle_suspected

# ==============================================================================
# 8. TEMPORAL SMOOTHER 
# ==============================================================================
_smoother = {"candidate": None, "count": 0, "stable": None, "type": ""}

def smooth_label(raw_label, is_symbol=False):
    s = _smoother
    threshold = SYMBOL_STABLE_FRAMES if is_symbol else ARROW_STABLE_FRAMES

    if raw_label == s["candidate"]:
        s["count"] += 1
    else:
        s["candidate"] = raw_label
        s["count"]     = 1
        s["type"]      = "symbol" if is_symbol else "arrow"

    if s["count"] >= threshold:
        s["stable"] = s["candidate"]

    return s["stable"]

def reset_smoother():
    _smoother["candidate"] = None
    _smoother["count"]     = 0
    _smoother["stable"]    = None
    _smoother["type"]      = ""

# ==============================================================================
# 9. THREAD 1 — LINE FOLLOWING (PID) + COLOUR SHORTCUT
# ==============================================================================
PID_ROI_TOP    = 141
PID_ROI_BOTTOM = 200
PID_ROI_MID    = (PID_ROI_TOP + PID_ROI_BOTTOM) // 2
FRAME_CENTER_X = 160

def line_thread():
    Kp = 0.6
    Ki = 0.0
    Kd = 1.0

    base_speed      = 35
    slow_speed      = 25   
    max_speed       = 70
    min_speed       = 0

    previous_error = 0
    integral       = 0
    COOLDOWN_DURATION = 2.0

    on_colour_shortcut   = False
    colour_shortcut_dir  = None
    shortcut_lost_frames = 0
    shortcut_entry_time  = 0.0   

    shortcut_lockout_until = 0.0

    arrow_exit_black_frames = 0
    ARROW_EXIT_BLACK_REQUIRED = 8   

    print("[LINE] Thread started.")
    erode_kernel = np.ones((5, 5), np.uint8)   

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
                
            on_colour_shortcut      = False
            colour_shortcut_dir     = None
            shortcut_lost_frames    = 0
            arrow_exit_black_frames = 0
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
            partial_sym      = shared["orb_partial_match"]

        effective_speed = slow_speed if partial_sym is not None else base_speed

        roi = bgr[PID_ROI_TOP:PID_ROI_BOTTOM, 0:320]

        pid_display = bgr.copy()
        cv2.rectangle(pid_display, (0, PID_ROI_TOP), (319, PID_ROI_BOTTOM), (200, 0, 200), 2)
        cv2.line(pid_display, (0, PID_ROI_MID), (319, PID_ROI_MID), (200, 0, 200), 1)

        main_mask   = cv2.inRange(roi, (0, 0, 0), (100, 100, 100))

        if now > shortcut_lockout_until:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            m1 = cv2.inRange(hsv_roi, (0,   100, 100), (10,  255, 255))
            m2 = cv2.inRange(hsv_roi, (160, 100, 100), (180, 255, 255))
            red_mask = cv2.bitwise_or(m1, m2)
            yellow_mask = cv2.inRange(hsv_roi, (15, 100, 100), (35, 255, 255))

            clean_red    = cv2.erode(red_mask,    erode_kernel, iterations=2)
            clean_yellow = cv2.erode(yellow_mask, erode_kernel, iterations=2)

            cnts_red, _    = cv2.findContours(clean_red,    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts_yellow, _ = cv2.findContours(clean_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            cnts_red = []
            cnts_yellow = []

        active_mask    = None
        detected_color = None
        contours_target = []
        tracking_label  = "BLACK"
        contours        = []

        if len(cnts_red) > 0 and len(cnts_yellow) > 0:
            max_r = cv2.contourArea(max(cnts_red,    key=cv2.contourArea))
            max_y = cv2.contourArea(max(cnts_yellow, key=cv2.contourArea))
            if max_r > max_y:
                contours_target, detected_color, active_mask = cnts_red,    "RED",    clean_red
            else:
                contours_target, detected_color, active_mask = cnts_yellow, "YELLOW", clean_yellow
        elif len(cnts_red)    > 0:
            contours_target, detected_color, active_mask = cnts_red,    "RED",    clean_red
        elif len(cnts_yellow) > 0:
            contours_target, detected_color, active_mask = cnts_yellow, "YELLOW", clean_yellow

        if active_mask is not None:
            tracking_label       = f"{detected_color} (Shortcut)"
            contours             = contours_target
            shortcut_lost_frames = 0

            if not on_colour_shortcut:
                c_target              = max(contours_target, key=cv2.contourArea)
                x_t, _, w_t, _        = cv2.boundingRect(c_target)
                entry_error           = int(x_t + w_t / 2) - FRAME_CENTER_X
                colour_shortcut_dir   = "RIGHT" if entry_error >= 0 else "LEFT"
                on_colour_shortcut    = True
                shortcut_entry_time   = now   
                with shared_lock:
                    shared["on_shortcut"]   = True
                    shared["shortcut_dir"]  = colour_shortcut_dir
                    shared["shortcut_type"] = "COLOUR"

            cv2.putText(pid_display,
                        f"SHORTCUT ({detected_color}) | Exit:{colour_shortcut_dir}",
                        (5, PID_ROI_TOP - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)

        if active_mask is None:
            active_mask    = cv2.erode(main_mask, erode_kernel, iterations=2)
            contours, _    = cv2.findContours(active_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if now < shortcut_lockout_until:
                tracking_label = f"BLACK (Lockout: {shortcut_lockout_until - now:.1f}s)"
            else:
                tracking_label = "BLACK (Main Route)"

            if on_colour_shortcut:
                shortcut_lost_frames += 1

                time_on_shortcut = now - shortcut_entry_time
                if shortcut_lost_frames > 12 and time_on_shortcut > 1.2:
                    if colour_shortcut_dir == "RIGHT":
                        motor_turn_right()
                    elif colour_shortcut_dir == "LEFT":
                        motor_turn_left()
                    set_pwm(ENL, 80); set_pwm(ENR, 80)
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
                        shared["shortcut_dir"]   = None
                        shared["shortcut_type"]  = None
                        shared["cooldown_until"] = time.time() + 1.0 
                    continue

            elif arrow_exit_armed:
                arrow_exit_black_frames += 1

                if arrow_exit_black_frames >= ARROW_EXIT_BLACK_REQUIRED:
                    if arrow_exit_dir == "RIGHT":
                        motor_turn_right()
                    elif arrow_exit_dir == "LEFT":
                        motor_turn_left()
                    set_pwm(ENL, 80); set_pwm(ENR, 80)
                    time.sleep(0.5)
                    motor_stop()

                    arrow_exit_black_frames = 0
                    with shared_lock:
                        shared["arrow_exit_armed"] = False
                        shared["arrow_exit_dir"]   = None
                        shared["on_shortcut"]      = False
                        shared["shortcut_dir"]     = None
                        shared["shortcut_type"]    = None
                        shared["cooldown_until"]   = time.time() + 1.0
                    previous_error = 0
                    integral       = 0
                    continue
            else:
                arrow_exit_black_frames = 0

        if len(contours) == 0:
            set_pwm(ENL, 100); set_pwm(ENR, 100)
            motor_turn_right() if previous_error > 0 else motor_turn_left()
            with shared_lock:
                shared["pid_display"]    = pid_display
                shared["tracking_label"] = tracking_label
            time.sleep(0.002)
            continue

        c       = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        line_cx = int(x + w / 2)
        error   = line_cx - FRAME_CENTER_X

        integral   += error
        derivative  = error - previous_error
        previous_error = error

        pid = Kp * error + Ki * integral + Kd * derivative

        left_speed  = max(min_speed, min(max_speed, effective_speed + pid))
        right_speed = max(min_speed, min(max_speed, effective_speed - pid))

        set_pwm(ENL, left_speed)
        set_pwm(ENR, right_speed)
        motor_forward()

        roi_line_y = PID_ROI_TOP + (y + h // 2)
        cv2.line(pid_display, (FRAME_CENTER_X, 230), (line_cx, roi_line_y), (255, 0, 0), 3)
        cv2.circle(pid_display, (line_cx, roi_line_y), 6, (0, 0, 255), -1)

        spd_label = " [SLOW-detect]" if partial_sym else ""
        cv2.putText(pid_display, f"Track:{tracking_label}{spd_label}",
                    (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
        cv2.putText(pid_display,
                    f"err:{error:+d} pid:{pid:.1f} L:{left_speed:.0f} R:{right_speed:.0f}",
                    (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 180, 0), 1)

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
            partial_match_label = None
            partial_match_count = 0
            orb_debug     = "VISION PAUSED (COOLDOWN)"
            display_label = None
        else:
            vision_roi = frame[VISION_ROI_TOP:VISION_ROI_BOTTOM, 0:320]
            vision_hsv = hsv_frame[VISION_ROI_TOP:VISION_ROI_BOTTOM, 0:320]
            
            orb_sym, above_floor, orb_debug, partial_sym = run_orb(vision_roi, vision_hsv)

            if partial_sym is not None and partial_sym == partial_match_label:
                partial_match_count += 1
                if partial_match_count >= ORB_PARTIAL_FRAMES and orb_sym is None:
                    orb_sym = partial_sym   
                    orb_debug += " [PARTIAL-FIRE]"
            else:
                partial_match_label = partial_sym
                partial_match_count = 1 if partial_sym else 0

            with shared_lock:
                shared["orb_partial_match"] = partial_match_label if partial_match_count >= 3 else None

            arrow_dir         = None
            recycle_suspected = False

            if not on_shortcut and orb_sym is None:
                arrow_dir, bw_mask, found_boxes, recycle_suspected = detect_arrows(
                    vision_roi, frame, roi_y_offset=VISION_ROI_TOP)

            if recycle_suspected and orb_sym is None:
                orb_sym = "RECYCLE"   

            if orb_templates:
                final_label = (orb_sym if orb_sym
                               else (f"ARROW {arrow_dir.upper()}" if arrow_dir else None))
            else:
                final_label = f"ARROW {arrow_dir.upper()}" if arrow_dir else None

            is_symbol_detection = final_label is not None and not final_label.startswith("ARROW")
            smoother_key        = ("ARROW" if (final_label and final_label.startswith("ARROW"))
                                    else final_label)
            smooth_key     = smooth_label(smoother_key, is_symbol=is_symbol_detection)
            display_label  = final_label if smooth_key == "ARROW" else smooth_key

        if display_label is not None:
            with shared_lock:
                if not shared["symbol_action"]:
                    shared["symbol"]         = display_label
                    shared["current_action"] = display_label

        vision_display = frame.copy()

        cv2.rectangle(vision_display,
                      (CROP_X, CROP_Y), (CROP_X + CROP_SIZE, CROP_Y + CROP_SIZE),
                      (0, 165, 255), 1)   

        cv2.rectangle(vision_display,
                      (0, VISION_ROI_TOP), (319, VISION_ROI_BOTTOM),
                      (200, 0, 200), 2)
        cv2.line(vision_display,
                 (0, VISION_ROI_MID), (319, VISION_ROI_MID),
                 (200, 0, 200), 1)

        if on_shortcut:
            cv2.putText(vision_display, "ARROW DETECT: SUPPRESSED (ON SHORTCUT)",
                        (5, VISION_ROI_BOTTOM + 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (0, 200, 255), 1)

        if display_label:
            cv2.putText(vision_display, display_label,
                        (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        if partial_match_label:
            cv2.putText(vision_display,
                        f"Partial[{partial_match_label}] {partial_match_count}/{ORB_PARTIAL_FRAMES}",
                        (5, VISION_ROI_BOTTOM + 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (100, 200, 255), 1)

        cv2.putText(vision_display, orb_debug,
                    (4, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (140, 140, 140), 1)

        with shared_lock:
            shared["vision_display"] = vision_display

        time.sleep(0.016)

    print("[VISION] Thread stopped.")

# ==============================================================================
# 11. SCREEN 1 — Main Status Dashboard
# ==============================================================================
def build_main_display():
    with shared_lock:
        tracking    = shared.get("tracking_label",  "BLACK")
        action      = shared.get("current_action",  "IDLE")
        on_sc       = shared.get("on_shortcut",     False)
        sc_dir      = shared.get("shortcut_dir",    None)
        sc_type     = shared.get("shortcut_type",   None)
        pid_err     = shared.get("pid_error",       0)
        cooldown    = shared.get("cooldown_until",  0)
        in_action   = shared.get("symbol_action",   False)
        frame       = shared.get("frame",           None)
        partial_sym = shared.get("orb_partial_match", None)

    now      = time.time()
    dashboard = np.zeros((240, 320, 3), dtype=np.uint8)

    cv2.rectangle(dashboard, (0, 0), (320, 30), (50, 50, 50), -1)
    
    cv2.putText(dashboard, "AGV MAIN STATUS v10.4",
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
    
    if on_sc:
        cv2.putText(dashboard, f"SHORTCUT [{sc_type}] → exit:{sc_dir}",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 180), 2)
    else:
        cv2.putText(dashboard, "Shortcut: none",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (70, 70, 70), 1)

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
    <title>AGV Dashboard v10.4</title>
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
    <h1>AGV Dashboard v10.4 (Master)</h1>
    <div class="fix-note">
      <b>v10.4 Final:</b> 110x110 Crop for clean templates, Homography for full-screen search, and Action UI translation fully integrated.
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
        print("  AGV v10.4 Flask Server RUNNING")
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