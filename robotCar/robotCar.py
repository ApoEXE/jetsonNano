# SDA = pin.SDA_1
# SCL = pin.SCL_1
# SDA_1 = pin.SDA
# SCL_1 = pin.SCL
#BCM
'''
IN1_A = 17
IN2_A = 18
IN3_B = 22
IN4_B = 23
'''
'''
#BOARD
IN1_A = 11
IN2_A = 12
IN3_B = 15
IN4_B = 16
'''
#SYSFS
IN1_A = 50
IN2_A = 79
IN3_B = 194
IN4_B = 232
from adafruit_servokit import ServoKit
#import RPi.GPIO as GPIO
import board
import busio
import time
import sys
import os

print("Initializing Servos")
i2c_bus0=(busio.I2C(board.SCL_1, board.SDA_1))
print("Initializing ServoKit")
ENA = ServoKit(channels=16, i2c=i2c_bus0)
ENB = ServoKit(channels=16, i2c=i2c_bus0)
cam = ServoKit(channels=16, i2c=i2c_bus0)
'''
#GPIO setup
# Pin Setup:
# Board pin-numbering scheme
print(GPIO.getmode())
GPIO.setmode(GPIO.BOARD)
# set pin as an output pin with optional initial state of HIGH
GPIO.setup(IN1_A, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(IN2_A, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(IN3_B, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(IN4_B, GPIO.OUT, initial=GPIO.HIGH)

def forward():
     GPIO.output(IN1_A, GPIO.HIGH)
     GPIO.output(IN2_A, GPIO.LOW)
'''
def sweepTest():
	sweep = range(0,180)
	for degree in sweep :
		cam.servo[15].angle=degree
    	# kit.servo[1].angle=degree
    	# time.sleep(0.01)

	time.sleep(0.5)
	sweep = range(180,0, -1)
	for degree in sweep :
		cam.servo[15].angle=degree



def main():
    # On the Jetson Nano
    # Bus 0 (pins 28,27) is board SCL_1, SDA_1 in the jetson board definition file
    # Bus 1 (pins 5, 3) is board SCL, SDA in the jetson definition file
    # Default is to Bus 1; We are using Bus 0, so we need to construct the busio first ...
    angle = int(sys.argv[1])
    #forward()
    # kit[0] is the bottom servo
    # kit[1] is the top servo

    os.system("echo "+str(IN1_A)+" > /sys/class/gpio/export")
    os.system("echo "+str(IN2_A)+" > /sys/class/gpio/export")
    os.system("echo "+str(IN3_B)+" > /sys/class/gpio/export")
    os.system("echo "+str(IN4_B)+" > /sys/class/gpio/export")
    os.system("echo out > /sys/class/gpio/gpio"+str(IN1_A)+"/direction")
    os.system("echo out > /sys/class/gpio/gpio"+str(IN2_A)+"/direction")
    os.system("echo out > /sys/class/gpio/gpio"+str(IN3_B)+"/direction")
    os.system("echo out > /sys/class/gpio/gpio"+str(IN4_B)+"/direction")
    print("Done initializing")
    
    ENB.servo[14].angle=angle #power on ENB
    ENA.servo[13].angle=angle #power on ENA
    
    os.system("echo 1 > /sys/class/gpio/gpio"+str(IN1_A)+"/value")
    os.system("echo 0 > /sys/class/gpio/gpio"+str(IN2_A)+"/value")
    os.system("echo 1 > /sys/class/gpio/gpio"+str(IN3_B)+"/value")
    os.system("echo 0 > /sys/class/gpio/gpio"+str(IN4_B)+"/value")
    sweepTest()

    #try:
        #while True:
            #sweepTest()

    #finally:
        #pass


if __name__ == '__main__':
    main()
   # GPIO.cleanup()