import cv2
from picamera2 import Picamera2
import time

print("Initializing Camera in VS Code...")

try:
    # 1. Start the camera
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    
    print("Camera warmed up! Press 'q' in the video window to quit.")
    time.sleep(2) # Give the sensor time to adjust to the light

    # 2. Start the loop
    while True:
        # Grab the frame and fix the Smurf/Blue skin issue
        raw_frame = picam2.capture_array()
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)

        # Show the frame
        cv2.imshow("VS Code Camera Test", frame)

        # Wait for the 'q' key to be pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # 3. Always release the camera!
    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera safely closed.")