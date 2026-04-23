# ==============================================================================
# W3_Track_v2 — 30 FPS Parallel Circuit
# ==============================================================================
#
# LECTURER REQUIREMENTS APPLIED:
#   - Shapes removed (arrows and symbols only — no diamonds, stars, etc.)
#   - Dynamic floor colour tracking (Target Colour → fallback to black line)
#   - HSV colour verification to distinguish BIOHAZARD (orange) vs BUTTON (other)
#   - "Not Symbol → Arrow" logical fallback (Branch C)
#
# CRITICAL FIXES APPLIED:
#   - Restored AGV_Combined3.py Vision Logic: ORB_MIN_FLOOR = 5, Full-screen Otsu Arrows.
#   - Shrunken Track ROI: Y-length heavily reduced to only look right in front of the robot.
#   - PID.py Display Style: Restored the clean, large blue error text and mask window.
#   - Visual Re-arming: Smart flag pauses vision until paper clears the camera.
#   - Shortcut Memory: Remembers entry side to smoothly exit Red/Yellow tracks.

import cv2           
import numpy as np   
import os            
import time          
import threading     
from picamera2 import Picamera2  
import pigpio        


# ==============================================================================
# 1. SHARED STATE — The "Parallel Circuit Message Board"
# ==============================================================================
shared_lock = threading.Lock()   

shared = {
    "line_display":   None,   
    "vision_display": None,   
    "mask_display":   None,   
    "track_mask_display": None,  # <--- 4TH DISPLAY FOR TRACK B&W MASK
    "frame":          None,   
    "symbol":         None,   
    "symbol_action":  False,  
    "waiting_for_clear_view": False, # <--- Smart Visual Re-arming flag
    "running":        True,   
    "key_press":      None,   
}


# ==============================================================================
# 2. CAMERA SETUP
# ==============================================================================
print("[INIT] Starting camera at 30 FPS / 320x240...")
picam2 = Picamera2()   

config = picam2.create_video_configuration(
    main={"size": (320, 240), "format": "BGR888"},  
    controls={"FrameRate": 30}                      
)
picam2.configure(config)
picam2.start()   

picam2.set_controls({
    "AwbEnable":    False,       
    "AeEnable":     False,        
    "FrameRate":    30,          
    "ExposureTime": 22000,       
    "AnalogueGain": 4.0,         
    "ColourGains":  (1.8, 1.2),  
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
    for pin in [IN1, IN2, IN3, IN4]:
        pi.write(pin, 0)      
    set_pwm(ENL, 0)           
    set_pwm(ENR, 0)           

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
    print(">>> BIOHAZARD/BUTTON: Stopping")
    motor_stop()
    time.sleep(1)   

def action_qr_code():
    print(">>> QR/FINGERPRINT: Biometric")
    motor_stop()
    time.sleep(1)   

def action_360_turn():
    print(">>> RECYCLE: Executing 360 Turn")
    pi.write(IN1, 0); pi.write(IN2, 1)   
    pi.write(IN3, 1); pi.write(IN4, 0)   
    set_pwm(ENL, 80); set_pwm(ENR, 80)   
    time.sleep(1.2)                      
    motor_stop()

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

SYMBOL_ACTIONS = {
    "BIOHAZARD":   action_stop,
    "BUTTON":      action_stop,
    "QR_CODE":     action_qr_code,
    "FINGERPRINT": action_qr_code,
    "RECYCLE":     action_360_turn,
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
# 5. ORB TEMPLATES & DETECTION SETUP
# ==============================================================================
TEMPLATE_DIR = "./orb_templates"   
ORB_SYMBOLS = ["BIOHAZARD", "RECYCLE", "QR_CODE", "FINGERPRINT", "BUTTON"]

ORB_RATIO_TEST = 0.75
ORB_MIN_FLOOR = 5     
ORB_GAP_FACTOR = 2.0

CROP_SIZE = 150                                    
CROP_X = (320 - CROP_SIZE) // 2                
CROP_Y = (240 - CROP_SIZE) // 2                

STABLE_FRAMES_REQUIRED = 4   
SWITCH_FRAMES_REQUIRED = 4   

os.makedirs(TEMPLATE_DIR, exist_ok=True)   

orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
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
    print(f"  [ORB] Saved: {symbol}")
    load_templates() 


# ==============================================================================
# 6. VISION MATH: ORB & ARROW DETECTION (Restored from AGV_Combined3.py)
# ==============================================================================

def run_orb(bgr_frame, hsv_frame):
    if not orb_templates:
        return None, False, "ORB: no templates"

    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    kp_live, des_live = orb.detectAndCompute(gray, None)
    live_kp = len(kp_live) if kp_live else 0   

    if des_live is None or len(kp_live) == 0:
        return None, False, "KP:0 | blind"

    results = []   
    for sym, (kp_tmpl, des_tmpl, _) in orb_templates.items():
        try:
            matches = bf.knnMatch(des_tmpl, des_live, k=2)
            good = [
                m for pair in matches if len(pair) == 2
                for m, n in [pair] if m.distance < ORB_RATIO_TEST * n.distance
            ]
            results.append((sym, len(good)))
        except Exception:
            results.append((sym, 0))   

    results.sort(key=lambda x: x[1], reverse=True)

    best_sym     = results[0][0] if results else None  
    best_count   = results[0][1] if results else 0      
    second_count = results[1][1] if len(results) >= 2 else 0 

    above_floor = best_count >= ORB_MIN_FLOOR
    clear_gap = (len(results) < 2) or (best_count >= ORB_GAP_FACTOR * second_count)
    confident = above_floor and clear_gap

    # --- HSV BIOHAZARD VS BUTTON VERIFICATION ---
    if confident and best_sym in ["BIOHAZARD", "BUTTON"]:
        h, w = bgr_frame.shape[:2]
        center_hue = np.median(hsv_frame[h//2 - 10 : h//2 + 10, w//2 - 10 : w//2 + 10, 0])
        if 10 <= center_hue <= 35:
            best_sym = "BIOHAZARD"
        else:
            best_sym = "BUTTON"

    if not above_floor:
        flag = f"FLOOR<{ORB_MIN_FLOOR}"          
    elif not clear_gap:
        flag = f"GAP"                             
    else:
        flag = "OK"                               

    debug_str = f"KP:{len(kp_live)} | BEST:{best_sym} {best_count}m [{flag}]"
    return (best_sym if confident else None), True, debug_str


geo_kernel = np.ones((5, 5), np.uint8)

def detect_arrows(bgr_frame, display_frame):
    """
    Arrow detection restored directly from AGV_Combined3.py
    """
    tally = {}

    yuv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YUV)
    y_channel = yuv[:, :, 0]
    
    blurred_y = cv2.GaussianBlur(y_channel, (7, 7), 0)

    _, cleaned = cv2.threshold(blurred_y, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    black_track_mask = cv2.inRange(bgr_frame, (0, 0, 0), (100, 100, 100))
    cleaned = cv2.bitwise_and(cleaned, cv2.bitwise_not(black_track_mask))

    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  geo_kernel)
    cleaned = cv2.morphologyEx(cleaned,  cv2.MORPH_CLOSE, geo_kernel)

    cnts, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 800 or area > 12000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0 or w == 0 or w > 200 or h > 200:
            continue

        aspect_ratio = float(w) / h
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            continue 

        area_ratio = area / float(w * h)
        if area_ratio > 0.65:
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

        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display_frame, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        tally[label] = tally.get(label, 0) + 1

    arrow_dir = None
    for name in tally:
        if "Arrow" in name:
            arrow_dir = name.split("(")[1].strip(")")
            break

    return arrow_dir, cleaned


# ==============================================================================
# TEMPORAL SMOOTHER & MEMORY WIPER
# ==============================================================================
_candidate_label = None   
_candidate_count = 0      
_stable_label    = None   

def smooth_label(raw_label):
    global _candidate_label, _candidate_count, _stable_label

    if raw_label == _candidate_label:
        _candidate_count += 1    
    else:
        _candidate_label = raw_label
        _candidate_count = 1

    threshold = SWITCH_FRAMES_REQUIRED if _stable_label is not None else STABLE_FRAMES_REQUIRED
    if _candidate_count >= threshold:
        _stable_label = _candidate_label   

    return _stable_label   

def reset_smoother():
    global _candidate_label, _candidate_count, _stable_label
    _candidate_label = None
    _candidate_count = 0
    _stable_label = None


# ==============================================================================
# 7. THREAD 1 — LINE FOLLOWING (PID)
# ==============================================================================
def line_thread():
    
    Kp = 0.5   
    Ki = 0.0   
    Kd = 1.0   

    base_speed = 40   
    max_speed  = 80   
    min_speed  = 0     

    previous_error = 0   
    integral       = 0   

    # >>> TEST DAY CONFIGURATION <<<
    TARGET_FLOOR_COLOR = "YELLOW"

    on_shortcut = False
    shortcut_dir = None
    shortcut_lost_frames = 0  

    print("[LINE] Thread started.")

    while True:
        with shared_lock:
            if not shared["running"]:
                break

        rgb = picam2.capture_array()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        display_bgr = bgr.copy()   

        with shared_lock:
            shared["frame"] = bgr.copy()

        now = time.time()
        with shared_lock:
            symbol    = shared["symbol"]         
            in_action = shared["symbol_action"]  

        if symbol is not None and not in_action:
            motor_stop()   

            with shared_lock:
                shared["symbol_action"] = True

            execute_symbol_action(symbol)  

            with shared_lock:
                shared["symbol"]         = None
                shared["symbol_action"]  = False
                # Smart Re-arming: Tells vision thread to wait until paper clears
                shared["waiting_for_clear_view"] = True 

            previous_error  = 0   
            integral        = 0
            continue   

        # --- HEAVILY SHRUNKEN TRACK ROI ---
        # Greatly reduced the Y-length! Now looks ONLY at the bottom band (Y: 180 to 240)
        roi_x1, roi_x2 = 60, 260
        roi_y1, roi_y2 = 180, 240  # <--- Changed from 140 to 180
        roi_w = roi_x2 - roi_x1
        roi_center = roi_w // 2  # Center of the 200px wide box = 100

        roi = bgr[roi_y1:roi_y2, roi_x1:roi_x2]
        
        main_mask = cv2.inRange(roi, (0, 0, 0), (100, 100, 100))

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        m_red1 = cv2.inRange(hsv_roi, (0,   50, 40), (10,  255, 255))
        m_red2 = cv2.inRange(hsv_roi, (160, 50, 40), (180, 255, 255))
        m_yellow = cv2.inRange(hsv_roi, (15, 50, 40), (35, 255, 255))
        
        shortcut_mask = cv2.bitwise_or(m_red1, m_red2)
        shortcut_mask = cv2.bitwise_or(shortcut_mask, m_yellow)

        active_mask    = None

        clean_target = cv2.erode(shortcut_mask, np.ones((5, 5), np.uint8), iterations=2)
        contours_target, _  = cv2.findContours(clean_target, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours_target) > 0:
            active_mask    = clean_target
            contours = contours_target
            shortcut_lost_frames = 0 

            if not on_shortcut:
                c_target = max(contours_target, key=cv2.contourArea)
                x_t, y_t, w_t, h_t = cv2.boundingRect(c_target)
                entry_error = int(x_t + (w_t / 2)) - roi_center
                shortcut_dir = "RIGHT" if entry_error >= 0 else "LEFT"
                on_shortcut = True

        if active_mask is None:
            active_mask    = cv2.erode(main_mask, np.ones((5, 5), np.uint8), iterations=2)
            contours, _    = cv2.findContours(active_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if on_shortcut:
                shortcut_lost_frames += 1
                if shortcut_lost_frames > 2: 
                    if shortcut_dir == "RIGHT":
                        pi.write(IN1, 1); pi.write(IN2, 0)
                        pi.write(IN3, 0); pi.write(IN4, 1)
                    elif shortcut_dir == "LEFT":
                        pi.write(IN1, 0); pi.write(IN2, 1)
                        pi.write(IN3, 1); pi.write(IN4, 0)
                    
                    set_pwm(ENL, 70); set_pwm(ENR, 70)
                    time.sleep(0.5) 
                    motor_stop()

                    on_shortcut = False
                    shortcut_dir = None
                    shortcut_lost_frames = 0
                    previous_error = 0
                    integral = 0
                    continue 

        if len(contours) == 0:
            set_pwm(ENL, 100); set_pwm(ENR, 100)
            if previous_error > 0: motor_turn_right()   
            else: motor_turn_left()    
            with shared_lock: 
                shared["line_display"] = display_bgr
                shared["track_mask_display"] = active_mask if active_mask is not None else main_mask
            time.sleep(0.005)
            continue   

        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        error = int(x + (w / 2)) - roi_center

        integral  += error                   
        derivative = error - previous_error  
        previous_error = error               

        pid = Kp * error + Ki * integral + Kd * derivative

        left_speed  = max(min_speed, min(max_speed, base_speed + pid))
        right_speed = max(min_speed, min(max_speed, base_speed - pid))

        set_pwm(ENL, left_speed); set_pwm(ENR, right_speed)
        motor_forward()  

        # =========================================================
        # --- PID.py STYLE DISPLAY LOGIC ---
        # =========================================================
        target_x = roi_x1 + int(x + w / 2)
        target_y = roi_y1 + int(y)

        # Draw the ROI boundary so you can see the shrunken zone
        cv2.rectangle(display_bgr, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 255), 2)
        
        # 1. Thick Blue Line 
        cv2.line(display_bgr, (160, 240), (target_x, target_y), (255, 0, 0), 3)
        
        # 2. Large Blue Error Number
        cv2.putText(display_bgr, str(error), (140, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if on_shortcut:
            cv2.putText(display_bgr, f"Shortcut: {shortcut_dir}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        with shared_lock:
            shared["line_display"] = display_bgr   
            shared["track_mask_display"] = active_mask if active_mask is not None else main_mask

        time.sleep(0.005)

    motor_stop()
    print("[LINE] Thread stopped.")


# ==============================================================================
# 8. THREAD 2 — SYMBOL & ARROW DETECTION 
# ==============================================================================
def vision_thread():
    capture_mode = False   
    clear_frames = 0  
    print("[VISION] Thread started.")

    while True:
        with shared_lock:
            if not shared["running"]: break
            frame_bgr = shared["frame"]    
            key_press = shared["key_press"]  
            shared["key_press"] = None       

        if frame_bgr is None:
            time.sleep(0.01)
            continue

        frame = frame_bgr.copy()
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
            time.sleep(0.03) 
            continue   

        # =======================================================================
        # DETECTION HIERARCHY & VISUAL RE-ARMING
        # =======================================================================
        with shared_lock:
            in_action = shared["symbol_action"]
            waiting_for_clear = shared.get("waiting_for_clear_view", False)

        if in_action:
            reset_smoother()  
            display_label = None
            orb_debug = "VISION PAUSED (MID-ACTION)"
            bw_mask = np.zeros((240, 320), dtype=np.uint8)
        else:
            orb_sym, orb_active, orb_debug = run_orb(frame, hsv_frame)

            if orb_sym:
                arrow_dir = None
                bw_mask   = np.zeros((240, 320), dtype=np.uint8)
            else:
                arrow_dir, bw_mask = detect_arrows(frame, frame)

            if orb_templates:
                raw_label = orb_sym if orb_sym else (f"ARROW {arrow_dir.upper()}" if arrow_dir else None)
            else:
                raw_label = f"ARROW {arrow_dir.upper()}" if arrow_dir else None

            # --- SMART VISUAL RE-ARMING ---
            if waiting_for_clear:
                display_label = None
                orb_debug = "WAITING FOR PAPER TO CLEAR..."
                
                if raw_label is None:
                    clear_frames += 1
                    if clear_frames >= 5:
                        with shared_lock:
                            shared["waiting_for_clear_view"] = False
                        clear_frames = 0
                else:
                    clear_frames = 0
            else:
                smoother_key = "ARROW" if (raw_label and raw_label.startswith("ARROW")) else raw_label
                smooth_key = smooth_label(smoother_key)
                display_label = raw_label if smooth_key == "ARROW" else smooth_key

        if display_label is not None:
            with shared_lock:
                if not shared["symbol_action"]:
                    shared["symbol"] = display_label

        if display_label:
            cv2.putText(frame, display_label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.putText(frame, orb_debug, (4, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)

        with shared_lock:
            shared["vision_display"] = frame     
            shared["mask_display"]   = bw_mask   

        time.sleep(0.03)

    print("[VISION] Thread stopped.")


# ==============================================================================
# 9. MAIN THREAD — GUI and Keyboard Handling
# ==============================================================================
if __name__ == "__main__":
    t_line   = threading.Thread(target=line_thread,   name="LINE",   daemon=True)
    t_vision = threading.Thread(target=vision_thread, name="VISION", daemon=True)

    t_line.start()    
    t_vision.start()  

    try:
        print("[MAIN] GUI Loop started. Displaying 4 windows.")
        while True:
            with shared_lock:
                if not shared["running"]: break
                l_disp = shared["line_display"]
                v_disp = shared["vision_display"]
                m_disp = shared["mask_display"]
                t_mask = shared.get("track_mask_display", None) # Grab PID Mask

            if v_disp is not None: cv2.imshow("1. AGV Vision", v_disp)
            if m_disp is not None: cv2.imshow("2. Vision Mask", m_disp)
            if l_disp is not None: cv2.imshow("3. Line Tracking", l_disp)
            
            # The 4th window will be a smaller, shrunken rectangle!
            if t_mask is not None: cv2.imshow("4. Track ROI Mask", t_mask) 

            key = cv2.waitKey(10) & 0xFF

            if key in (ord('q'), ord('Q')):
                with shared_lock: shared["running"] = False
                break
            elif key != 255:
                with shared_lock: shared["key_press"] = key

    except KeyboardInterrupt:
        print("\n[MAIN] Keyboard interrupt — stopping.")
        with shared_lock: shared["running"] = False

    finally:
        t_line.join(timeout=2)    
        t_vision.join(timeout=2)  
        motor_stop()             
        picam2.stop()            
        cv2.destroyAllWindows()  
        pi.stop()                
        print("[MAIN] Shutdown complete.")