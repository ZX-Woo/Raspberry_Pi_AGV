import cv2
import numpy as np
from picamera2 import Picamera2, Preview
import time
import pigpio
import math

# --- Camera Setup ---
picam2 = Picamera2()

preview_config = picam2.create_preview_configuration(
    # Using YUV420 is much lighter on the Pi's memory
    main={"size": (320, 240), "format": "RGB888"}, # SCALED DOWN TO 320x240
    controls={
        "FrameRate": 60,
        "AeEnable": True,
        "AwbEnable": False,
        "ColourGains": (1.5, 1.2)
#        "ExposureTime": 22000,
#        "AnalogueGain": 5.0
    }
)
picam2.configure(preview_config)
picam2.start()
time.sleep(0.01) 

# --- Motor Pins & Pigpio Setup ---
ENL, IN1, IN2 = 13, 5, 6
ENR, IN3, IN4 = 12, 19, 26
pi = pigpio.pi()

# SCALED PID: Because the screen is half as wide, the 'error' is half as large.
# Doubling Kp and Kd keeps your steering strength exactly the same!
Kp = 1.0 # Was 0.4
Ki = 0.0
Kd = 1.0 # Was 0.4

x, y, w, h = 160, 0, 0, 0 # Default X scaled from 360 to 160
previous_error = 0
integral = 0
last_time = time.time()

base_speed = 40 #45
max_speed = 50 #80
min_speed = 0
for pin in [IN1, IN2, IN3, IN4, ENL, ENR]:
    pi.set_mode(pin, pigpio.OUTPUT)

def set_pwm(pin, duty_percent):
    pi.set_PWM_dutycycle(pin, int(255 * duty_percent / 100))

def stop():
    for pin in [IN1, IN2, IN3, IN4]: pi.write(pin, 0)
    set_pwm(ENL, 0); set_pwm(ENR, 0)

# Standard motor functions
def forward():
    pi.write(IN1, 0); pi.write(IN2, 1); pi.write(IN3, 0); pi.write(IN4, 1)
def turn_right():
    pi.write(IN1, 1); pi.write(IN2, 0); pi.write(IN3, 0); pi.write(IN4, 1)
def turn_left():
    pi.write(IN1, 0); pi.write(IN2, 1); pi.write(IN3, 1); pi.write(IN4, 0)

#Detect shapes
def detect_shape(cnt, thresh_frame):
    shape = ""
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    area = cv2.contourArea(cnt)
    M = cv2.moments(cnt)

    if M["m00"] == 0: return "" # Prevent division by zero
    cx = int(M["m10"] / M["m00"]) 
    cy = int(M["m01"] / M["m00"]) 
    
    # Check if the center point is actually on the arrow color
    if cy >= thresh_frame.shape[0] or cx >= thresh_frame.shape[1]: return ""
    
    circularity = (4 * math.pi * area) / (peri * peri)
    
    if thresh_frame[cy, cx] == 255:
        maxdist = 0
        tip_idx = 0
        for i in range(len(approx)):
            dist = np.linalg.norm(approx[i][0] - (cx, cy))
            if dist > maxdist:
                maxdist = dist
                tip_idx = i
        
        # Calculate angle
        tip_ang = math.atan2(cy - approx[tip_idx][0][1], approx[tip_idx][0][0] - cx) * 180 / math.pi
        arrow_ang = (tip_ang - 90) % 360
        
        # --- DEBUG PRINT HERE ---
        # This tells you: the raw angle found, and how "circular" the arrow looks
        print(f"DEBUG: Ang={arrow_ang:.1f} | Circ={circularity:.2f} | Pts={len(approx)}")

        if 45 <= arrow_ang < 135:
            shape = "arrow (right)"
            if circularity > 0.5: shape = "arrow (left)"
        elif 135 <= arrow_ang < 225:
            shape = "arrow (up)"
            if circularity > 0.5: shape = "arrow (down)"
        elif 225 <= arrow_ang < 315:
            shape = "arrow (left)"
            if circularity > 0.5: shape = "arrow (right)"
        else:
            shape = "arrow (down)"
            if circularity > 0.5: shape = "arrow (up)"
            
    return shape # Moved outside the vertex loop so it actually returns

try:
    while True:
        # 1. Capture in YUV
        yuv_frame = picam2.capture_array()
        image = yuv_frame
        
        # SCALED ROI: Old was [260:360, 0:640]. New is divided by 2.
        roi = image[130:180, 0:320] 
        kernel = np.ones((5, 5), np.uint8)
        
        thresh_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        HSVimage = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # SCALED PATCH: Old was [40:60, 300:340]. New is divided by 2.
        # patch = HSVimage[20:30, 150:170]  # centre of ROI
        # print(f"Yellow patch H={np.median(patch[:,:,0]):.1f} S={np.median(patch[:,:,1]):.1f} V={np.median(patch[:,:,2]):.1f}")
        
        #for arrows
        redthresh_frame1 = cv2.inRange(thresh_frame,(136,80,80),(180,255,255))
        redthresh_frame2=cv2.inRange(thresh_frame,(0,80,80),(15,255,255))
        redthresh_frame=cv2.bitwise_or(redthresh_frame1,redthresh_frame2)
        #greenthresh_frame=cv2.inRange(thresh_frame,(50,85,40),(90,255,255))
        greenthresh_frame = cv2.inRange(thresh_frame, (60, 100, 100), (90, 255, 255))
        bluethresh_frame=cv2.inRange(thresh_frame,(100,80,2),(135,255,255))

        #for line detection
        Blackline=cv2.inRange(roi,(0,0,0),(100,100, 100))
        Redline1=cv2.inRange(HSVimage,(136,80,80),(180,255,255))
        Redline2=cv2.inRange(HSVimage,(0,80,80),(15,255,255))
        Redline=cv2.bitwise_or(Redline1,Redline2)
        
        Yellowline = cv2.inRange(HSVimage, (15, 100, 100), (35, 255, 255))
        
        Redline=cv2.erode(Redline,kernel,iterations=2)
        Yellowline=cv2.erode(Yellowline,kernel,iterations=2)
        Blackline=cv2.erode(Blackline,kernel,iterations=2)

        #finding contours and hierarchies for line detection
        blackcontours, blackhierarchy = cv2.findContours(Blackline, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        redcontours, redhierarchy = cv2.findContours(Redline, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        yellowcontours, yellowhierarchy = cv2.findContours(Yellowline, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #clean up images for arrows/symbols. Add erosion if needed !
        redthresh_frame = cv2.GaussianBlur(redthresh_frame, (5, 5), 0)
        greenthresh_frame = cv2.GaussianBlur(greenthresh_frame, (5, 5), 0)
        bluethresh_frame = cv2.GaussianBlur(bluethresh_frame, (5, 5), 0)

        #applying canny edge detection for arrows/symbols.
        rededges = cv2.Canny(redthresh_frame, 30, 100)
        greenedges = cv2.Canny(greenthresh_frame, 30, 100)
        blueedges = cv2.Canny(bluethresh_frame, 30, 100)

        #finding contours and hierarchies for arrows/symbols
        red_symbol_contours, _ = cv2.findContours(rededges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        green_symbol_contours, _ = cv2.findContours(greenedges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        blue_symbol_contours, _ = cv2.findContours(blueedges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        shape = ""
        # SCALED AREA: 8000 divided by 4 = 2000
        for cnt in red_symbol_contours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                shape = detect_shape(cnt, redthresh_frame)
        if shape !="":
            print(shape)

        for cnt in green_symbol_contours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                shape = detect_shape(cnt, greenthresh_frame)
        if shape !="":
            print(shape)

        for cnt in blue_symbol_contours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                shape = detect_shape(cnt, bluethresh_frame)
        if shape !="":
            print(shape)

        if "left" in shape or "right" in shape:
            if "left" in shape:
                #adjust as needed based on testing. This is a strong correction because you want to turn sharply when you see a left arrow.
                set_pwm(ENL, 70)
                set_pwm(ENR, 70)
                turn_left()
                time.sleep(0.1)
                print("Turning left")

            elif "right" in shape:
                set_pwm(ENL, 70)
                set_pwm(ENR, 70)
                turn_right()
                time.sleep(0.1)
                print("Turning right")

            previous_error = 0
        else: 
        # SCALED DRAWING: (320, 430) becomes (160, 215). Y coord 225 becomes 155.
            if len(redcontours) > 0:
                c = max(redcontours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                cv2.line(image, (160, 215), (int(x + (w / 2)), 155), (255, 0, 0), 3)
                print(f"Red contour found at X={x + (w / 2)} | Error={int(x + (w / 2)) - 160}")

            elif len(yellowcontours) > 0:
                c = max(yellowcontours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                cv2.line(image, (160, 215), (int(x + (w / 2)), 155), (0, 255, 255), 3)
                print(f"Yellow contour found at X={x + (w / 2)} | Error={int(x + (w / 2)) - 160}")
            
            elif len(blackcontours) > 0:
                c = max(blackcontours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                cv2.line(image, (160, 215), (int(x + (w / 2)), 155), (0, 0, 255), 3)
            
            # SCALED CENTER: Image width is 320, so center is 160
            error = int(x + (w / 2)) - 160 
            cv2.putText(image, str(error), (140, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(image, f"R:{len(redcontours)} Y:{len(yellowcontours)} B:{len(blackcontours)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
            cv2.imshow("Contour", image)
            cv2.imshow("Red roi", Redline)
            cv2.imshow("Yellow roi", Yellowline)
            cv2.imshow("Black roi", Blackline)

            # --- PID Calculation ---
            current_time = time.time()
            dt = max(current_time - last_time, 0.01)
            last_time = current_time

            P = Kp * error
            integral += error * dt
            I = Ki * integral
            derivative = (error - previous_error) / dt if dt > 0 else 0
            D = Kd * derivative

            pid = P + I + D
            previous_error = error

            left_speed = base_speed + pid
            right_speed = base_speed - pid

            left_speed = max(min_speed, min(max_speed, left_speed))
            right_speed = max(min_speed, min(max_speed, right_speed))

            set_pwm(ENL, left_speed)
            set_pwm(ENR, right_speed)

            forward()

        # Added. Remove if necessary!
        previous_error = error
        last_time = current_time
        # --- PID Calculation ---
        # integral += error;
        # derivative = error - previous_error;
        # previous_error = error;

        # pid = Kp * error + Ki * integral + Kd * derivative
        
        # left_speed = base_speed + pid
        # right_speed = base_speed - pid

        # left_speed = max(min_speed, min(max_speed, left_speed))
        # right_speed = max(min_speed, min(max_speed, right_speed))

        # set_pwm(ENL, left_speed)
        # set_pwm(ENR, right_speed)

        # forward()

        if len(redcontours) == 0 and len(yellowcontours) == 0 and len(blackcontours) == 0:
            if previous_error > 0:
                set_pwm(ENL, 60)
                set_pwm(ENR, 60)
                turn_right()
            else:
                set_pwm(ENL, 60)
                set_pwm(ENR, 60)
                turn_left()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    stop()
    picam2.stop()
    cv2.destroyAllWindows()
    pi.stop()