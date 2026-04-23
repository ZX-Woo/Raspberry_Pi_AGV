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
pwm_left = GPIO.PWM(ENL, 1000) #1Hz
pwm_right = GPIO.PWM(ENR, 1000)
pwm_left.start(0)
pwm_right.start(0)


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
        
        
try:
        while True:
            cmd = input("Enter f20, f40, f60, f80, f100, b20, b40, b60, b80, b100, l20, l40, l60, l80, l100, r20, r40, r60, r80, r100, s, q: ")
        
            if cmd == 'f20':
                pwm_left.ChangeDutyCycle(20)
                pwm_right.ChangeDutyCycle(20)
                forward()
            elif cmd == 'f40':
                pwm_left.ChangeDutyCycle(40)
                pwm_right.ChangeDutyCycle(40)
                forward()
            elif cmd == 'f60':
                pwm_left.ChangeDutyCycle(60)
                pwm_right.ChangeDutyCycle(60)
                forward()
            elif cmd == 'f80':
                pwm_left.ChangeDutyCycle(80)
                pwm_right.ChangeDutyCycle(80)
                forward()
            elif cmd == 'f100':
                pwm_left.ChangeDutyCycle(100)
                pwm_right.ChangeDutyCycle(100)
                forward()
            elif cmd == 'b20':
                pwm_left.ChangeDutyCycle(20)
                pwm_right.ChangeDutyCycle(20)
                backward()
            elif cmd == 'b40':
                pwm_left.ChangeDutyCycle(40)
                pwm_right.ChangeDutyCycle(40)
                backward()
            elif cmd == 'b60':
                pwm_left.ChangeDutyCycle(60)
                pwm_right.ChangeDutyCycle(60)
                backward()
            elif cmd == 'b80':
                pwm_left.ChangeDutyCycle(80)
                pwm_right.ChangeDutyCycle(80)
                backward()
            elif cmd == 'b100':
                pwm_left.ChangeDutyCycle(100)
                pwm_right.ChangeDutyCycle(100)
                backward()
            elif cmd == 'l20':
                pwm_left.ChangeDutyCycle(20)
                pwm_right.ChangeDutyCycle(20)
                turn_left()
            elif cmd == 'l40':
                pwm_left.ChangeDutyCycle(40)
                pwm_right.ChangeDutyCycle(40)
                turn_left()
            elif cmd == 'l60':
                pwm_left.ChangeDutyCycle(60)
                pwm_right.ChangeDutyCycle(60)
                turn_left()
            elif cmd == 'l80':
                pwm_left.ChangeDutyCycle(80)
                pwm_right.ChangeDutyCycle(80)
                turn_left()
            elif cmd == 'l100':
                pwm_left.ChangeDutyCycle(100)
                pwm_right.ChangeDutyCycle(100)
                turn_left()
            elif cmd == 'r20':
                pwm_left.ChangeDutyCycle(20)
                pwm_right.ChangeDutyCycle(20)
                turn_right()
            elif cmd == 'r40':
                pwm_left.ChangeDutyCycle(40)
                pwm_right.ChangeDutyCycle(40)
                turn_right()
            elif cmd == 'r60':
                pwm_left.ChangeDutyCycle(60)
                pwm_right.ChangeDutyCycle(60)
                turn_right()
            elif cmd == 'r80':
                pwm_left.ChangeDutyCycle(80)
                pwm_right.ChangeDutyCycle(80)
                turn_right()
            elif cmd == 'r100':
                pwm_left.ChangeDutyCycle(100)
                pwm_right.ChangeDutyCycle(100)
                turn_right()
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


