# import cv2 as cv
# import numpy as np
# img = cv.imread("Saved pictures/cat.jpg")
# cv.imshow("Cat", img)

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

# kernel = np.ones((5,5),np.uint8)
# erosion = cv.erode(gray, kernel, iterations = 1)
# cv.imshow('Eroded', erosion)
# cv.waitKey(0)
import cv2
import numpy as np
from picamera2 import Picamera2, Preview
import time
import pigpio

# --- Camera Setup ---
picam2 = Picamera2()
picam2.start_preview(Preview.QTGL) 

preview_config = picam2.create_preview_configuration(
    # Using YUV420 is much lighter on the Pi's memory
    main={"size": (640, 480), "format": "YUV420"}, 
    controls={
        "FrameRate": 50,
        "AeEnable": True,
#         "ExposureTime": 22000,
#         "AnalogueGain": 5.0
    }
)
picam2.configure(preview_config)
picam2.start()
time.sleep(0.01) 

# --- Motor Pins & Pigpio Setup ---
ENL, IN1, IN2 = 13, 5, 6
ENR, IN3, IN4 = 12, 19, 26
pi = pigpio.pi()
Kp = 0.4 #0.4
Ki = 0.0
Kd = 0.4

previous_error = 0
integral = 0
#last_time = time.time()

base_speed = 100 #45
max_speed = 100 #80
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

x, y, w, h = 360, 0, 0, 0 

try:
    while True:
        # 1. Capture in YUV
        yuv_frame = picam2.capture_array()
        
        # 2. Convert YUV to BGR for OpenCV
        # This is the "magic" line that prevents the Format Error
        image = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV420sp2BGR)
        
        # --- Your CV Logic ---
        roi = image[260:360, 0:640] 
        blackline = cv2.inRange(roi, (0, 0, 0), (100, 100, 100))
        kernel = np.ones((5, 5), np.uint8)
        blackline = cv2.erode(blackline, kernel, iterations=2) #remove background noise
        
        contours, _ = cv2.findContours(blackline, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.line(image, (320, 430), (int(x + (w / 2)), 225), (255, 0, 0), 3)
            
        error = int(x + (w / 2)) - 320 #object center - screen center
        cv2.putText(image, str(error), (280, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Contour", image)
        cv2.imshow("ROI View", blackline)
      

        # --- PID Calculation ---

        #P = Kp * error
        #integral += error * dt
        #I = Ki * integral
        #derivative = (error - previous_error) / dt if dt > 0 else 0
        #D = Kd * derivative
        integral += error;
        derivative = error - previous_error;
        previous_error = error;

        pid = Kp * error + Ki * integral + Kd * derivative
        

        left_speed = base_speed + pid
        right_speed = base_speed - pid

        left_speed = max(min_speed, min(max_speed, left_speed))
        right_speed = max(min_speed, min(max_speed, right_speed))

        set_pwm(ENL, left_speed)
        set_pwm(ENR, right_speed)

        forward()
      
        if len(contours) == 0:
    # Line lost ? turn in last known direction
            if previous_error > 0:
                # Line was on the right
                set_pwm(ENL, 100)
                set_pwm(ENR, 100)
                turn_right()
            else:
                # Line was on the left
                set_pwm(ENL, 100)
                set_pwm(ENR, 100)
                turn_left()
            continue

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    stop()
    picam2.stop()
    cv2.destroyAllWindows()
    pi.stop()




