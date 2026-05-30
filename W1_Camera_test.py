from picamera2 import Picamera2, Preview
import time

picam2 = Picamera2()
picam2.start_preview(Preview.QTGL)

preview_config = picam2.create_preview_configuration(
    main={"size":(720, 720), "format":"YUV420"},
    controls={"FrameRate":50,
              "AeEnable": False,
              "ExposureTime":22000,
              "AnalogueGain":5.0
              }
    )

picam2.configure(preview_config)

picam2.start()
time.sleep()