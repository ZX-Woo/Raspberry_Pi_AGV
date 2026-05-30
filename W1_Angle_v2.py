import RPi.GPIO as GPIO
import time
import math 

ENL = 13
IN1 = 5
IN2 = 6

ENR = 12
IN3 = 19
IN4 = 26

GPIO.setmode(GPIO.BCM)

motor_pins = [IN1, IN2, IN3, IN4]
for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)
    
GPIO.setup(ENL, GPIO.OUT)
GPIO.setup(ENR, GPIO.OUT)

#PWM SETUP (0-100 duty cycle)
pwm_left = GPIO.PWM(ENL, 1000) #1Hz
pwm_right = GPIO.PWM(ENR, 1000)
pwm_left.start(0)
pwm_right.start(0)

wheel_radius = 65/2

ANGLE_SPEED_MAP = {
    45: (50,70),
    75: (55,68),
    90: (65,68),
    180: (58,70),
    270: (55,65)
    }

def stop():
    for pin in motor_pins:
        GPIO.output(pin, GPIO.LOW)
        pwm_left.ChangeDutyCycle(0)
        pwm_right.ChangeDutyCycle(0)
        
def forward():
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
        
        
def backward():
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)


def turn_right():
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)
        
        
def turn_left():
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
        
def turn_time_for_angle(angle, speed, wheel_radius):
        return (angle * 3.14159 * wheel_radius) / (180 * speed)
        
def turn_right_angle(angle):
        if angle not in ANGLE_SPEED_MAP:
            print("Unsupported angle")
            return
        
        left_pwm, right_pwm = ANGLE_SPEED_MAP[angle]
        
        avg_speed = (left_pwm + right_pwm)/2
        turn_time = turn_time_for_angle(angle, avg_speed, wheel_radius)
        
        
        pwm_left.ChangeDutyCycle(left_pwm)
        pwm_right.ChangeDutyCycle(right_pwm)
        turn_right()
        time.sleep(turn_time)
        stop()

def turn_left_angle(angle):
        if angle not in ANGLE_SPEED_MAP:
            print("Unsupported angle")
            return
        
        left_pwm, right_pwm = ANGLE_SPEED_MAP[angle]
        
        avg_speed = (left_pwm + right_pwm)/2
        turn_time = turn_time_for_angle(angle, avg_speed, wheel_radius)
        
        
        pwm_left.ChangeDutyCycle(left_pwm)
        pwm_right.ChangeDutyCycle(right_pwm)
        turn_left()
        time.sleep(turn_time)
        stop()
        

try:
        while True:
            cmd = input("Enter r45, r75, r90, r180, r270, l45, l75, l90, l180, l270, s, q: ")
        
            if cmd == 'r90':
                turn_right_angle(90)
            elif cmd == 'r45':
                turn_right_angle(45)
            elif cmd == 'r180':
                turn_right_angle(180)
            elif cmd == 'r75':
                turn_right_angle(75)
            elif cmd == 'r270':
                turn_right_angle(270)
            elif cmd == 'l90':
                turn_left_angle(90)
            elif cmd == 'l45':
                turn_left_angle(45)
            elif cmd == 'l180':
                turn_left_angle(180)
            elif cmd == 'l75':
                turn_left_angle(75)
            elif cmd == 'l270':
                turn_left_angle(270)
            elif cmd == 's':
                stop()
            elif cmd == 'q':
                break
            else:
                print("Invalid command")
            

except KeyboardInterrupt:
    pass

finally:
    stop()
    pwm_left.stop()
    pwm_right.stop()
    GPIO.cleanup()
    print("Program exited safely")

