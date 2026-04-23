import RPi.GPIO as GPIO
import time

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
pwm_left = GPIO.PWM(ENL, 1050) 
pwm_right = GPIO.PWM(ENR, 1050)
pwm_left.start(0)
pwm_right.start(0)


def stop():
    for pin in motor_pins:
        GPIO.output(pin, GPIO.LOW)
        pwm_left.ChangeDutyCycle(0)
        pwm_right.ChangeDutyCycle(0)
        
def move(direction, travel_time):
    if direction == 'f':
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
        
        
    elif direction == 'b':
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)

    else:
        print("Invalid direction")
        return
    
    time.sleep(travel_time)
    stop()

#Velocity -> PWM
def velocity_to_pwm(v):
    if v == 30:
        return 50
    elif v == 40:
        return 60
    elif v == 50:
        return 80
    elif v == 60:
        return 100
    else:
        return None

#Main Loop        
try:
    while True:
        v = int(input("\nEnter velocity (30,40,50,60) cm/s or 0 to quit:"))
        
        if v == 0:
            break
        
        pwm = velocity_to_pwm(v)
        if pwm is None:
            print("Invalid velocity")
            continue
        
        pwm_left.ChangeDutyCycle(pwm)
        pwm_right.ChangeDutyCycle(pwm)
        
        direction = input("Direction (f = forward, b = backward):").lower()
        if direction not in ['f', 'b']:
            print("Invalid direction")
            stop()
            continue
        
        mode = input("Mode: (t) time-based or (d) distance-based:").lower()
        
        if mode == 't':
            t = float(input("Enter travel time(seconds):"))
            d = v * t 
            print(f"Estimated distance:{d:.2f}cm")
            move(direction, t)
        
        elif mode == 'd':
            d = float(input("Enter distance (cm):"))
            t = d/v
            print(f"Required time:{t:.2f}seconds")
            move(direction, t)
        
        else:
            print("Invalid mode")
            stop()
        
except KeyboardInterrupt:
    pass

finally:
    stop()
    pwm_left.stop()
    pwm_right.stop()
    GPIO.cleanup()
    print("Program exited safely")







