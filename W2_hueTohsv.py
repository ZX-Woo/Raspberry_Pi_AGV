import cv2
import numpy as np
from picamera2 import Picamera2

# Global variables to track where you click
clicked_x, clicked_y = -1, -1

# The function that fires every time you click the mouse
def mouse_click(event, x, y, flags, param):
    global clicked_x, clicked_y
    # If the left mouse button is clicked, save the X and Y coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_x, clicked_y = x, y

print("Initializing Live HSV Picker...")
picam2 = Picamera2()

# Using BGR888 configuration
config = picam2.create_video_configuration(main={"size": (320, 240), "format": "RGB888"})
picam2.configure(config)
picam2.start()

# Applying lighting calibration so the colors match your main code!
picam2.set_controls({"AwbEnable": False, "AeEnable": False, "FrameRate": 30})
picam2.set_controls({"ColourGains": (1.8, 1.2)}) 
picam2.set_controls({"AnalogueGain": 4.0}) 

# Create the window first, so we can attach the mouse clicker to it
cv2.namedWindow("Live HSV Picker")
cv2.setMouseCallback("Live HSV Picker", mouse_click)

print("\n>>> READY! Click anywhere on the video feed to get the exact HSV values! <<<")
print(">>> Press 'q' to quit. <<<\n")

try:
    while True:
        # Grab the frame and convert it to HSV
        frame = picam2.capture_array()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # If you clicked somewhere on the screen...
        if clicked_x != -1 and clicked_y != -1:
            # Extract the exact [Hue, Saturation, Value] array from that specific pixel
            current_hsv = hsv_frame[clicked_y, clicked_x]
            
            # Format the numbers into a readable string
            hsv_text = f"H: {current_hsv[0]} | S: {current_hsv[1]} | V: {current_hsv[2]}"
            
            # Print it directly to your terminal so you can copy/paste it later
            print(f"Target Acquired at (X:{clicked_x}, Y:{clicked_y}) -> {hsv_text}")
            
            # Draw a circle on the video feed exactly where you clicked
            cv2.circle(frame, (clicked_x, clicked_y), 5, (0, 255, 0), -1)
            
            # Slap the HSV numbers on the top left of the video feed
            cv2.putText(frame, hsv_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Reset the click variables so it waits for your next click
            clicked_x, clicked_y = -1, -1

        cv2.imshow("Live HSV Picker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("HSV Picker safely closed.")