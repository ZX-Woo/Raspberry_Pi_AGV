import pigpio
import time

# --- Motor Pins (BCM numbering) ---
ENL = 13
IN1 = 5
IN2 = 6

ENR = 12
IN3 = 19
IN4 = 26

# --- Connect to pigpio daemon ---
pi = pigpio.pi()
if not pi.connected:
    raise Exception("Could not connect to pigpio daemon. Run 'sudo pigpiod' first!")

# --- Set all pins as outputs ---
motor_pins = [IN1, IN2, IN3, IN4, ENL, ENR]
for pin in motor_pins:
    pi.set_mode(pin, pigpio.OUTPUT)

# --- PWM Helper (maps 0-100% to pigpio 0-255) ---
def set_pwm(pin, duty_percent):
    duty = int(255 * duty_percent / 100)
    pi.set_PWM_dutycycle(pin, duty)

# Initialize PWM at 0%
set_pwm(ENL, 0)
set_pwm(ENR, 0)

# --- Motor control functions ---
def stop():
    for pin in [IN1, IN2, IN3, IN4]:
        pi.write(pin, 0)
    set_pwm(ENL, 0)
    set_pwm(ENR, 0)

def forward():
    pi.write(IN1, 1)
    pi.write(IN2, 0)
    pi.write(IN3, 1)
    pi.write(IN4, 0)

def backward():
    pi.write(IN1, 0)
    pi.write(IN2, 1)
    pi.write(IN3, 0)
    pi.write(IN4, 1)

def turn_right():
    pi.write(IN1, 1)
    pi.write(IN2, 0)
    pi.write(IN3, 0)
    pi.write(IN4, 1)

def turn_left():
    pi.write(IN1, 0)
    pi.write(IN2, 1)
    pi.write(IN3, 1)
    pi.write(IN4, 0)

# --- Turn by angle (time-based) ---
def turn_right_angle(angle, speed=80):
    time_per_degree = 1/240.0  # adjust experimentally
    offsets = {
        45: -0.095,
        75: -0.055,
        90: -0.120,
        180: -0.160,
        270: -0.200
        }
    turn_time = angle * time_per_degree
    set_pwm(ENL, speed)
    set_pwm(ENR, speed)
    turn_right()
    time.sleep(turn_time-offsets.get(angle, 0))
    stop()

def turn_left_angle(angle, speed=80):
    time_per_degree = 1/200.0  # adjust experimentally
    offsets = {
        45: -0.060,
        75: -0.020,
        90: -0.070,
        180:-0.030,
        270: 0.025
        }
    turn_time = angle * time_per_degree
    set_pwm(ENL, speed)
    set_pwm(ENR, speed)
    turn_left()
    time.sleep(turn_time-offsets.get(angle, 0))
    stop()

# --- Main Loop ---
try:
    while True:
        cmd = input("Enter r45, r75, r90, r180, r270, l45, l75, l90, l180, l270, s, q: ").strip()
        
        if cmd == 'r45': turn_right_angle(45)
        elif cmd == 'r75': turn_right_angle(75)
        elif cmd == 'r90': turn_right_angle(90)
        elif cmd == 'r180': turn_right_angle(180)
        elif cmd == 'r270': turn_right_angle(270)
        elif cmd == 'l45': turn_left_angle(45)
        elif cmd == 'l75': turn_left_angle(75)
        elif cmd == 'l90': turn_left_angle(90)
        elif cmd == 'l180': turn_left_angle(180)
        elif cmd == 'l270': turn_left_angle(270)
        elif cmd == 's': stop()
        elif cmd == 'q': break
        else:
            print("Invalid command")

except KeyboardInterrupt:
    print("\nKeyboard interrupt received!")

finally:
    stop()
    pi.stop()
    print("Program exited safely")


