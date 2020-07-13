# SDA = pin.SDA_1
# SCL = pin.SCL_1
# SDA_1 = pin.SDA
# SCL_1 = pin.SCL

from adafruit_servokit import ServoKit
import board
import busio
import time
import sys
import os
print("Initializing Servos")
i2c_bus0=(busio.I2C(board.SCL_1, board.SDA_1))
print("Initializing ServoKit")
kit = ServoKit(channels=16, i2c=i2c_bus0)

def sweepTest():
	sweep = range(0,180)
	for degree in sweep :
		kit.servo[0].angle=degree
    	# kit.servo[1].angle=degree
    	# time.sleep(0.01)

	time.sleep(0.5)
	sweep = range(180,0, -1)
	for degree in sweep :
		kit.servo[0].angle=degree



def main():
    # On the Jetson Nano
    # Bus 0 (pins 28,27) is board SCL_1, SDA_1 in the jetson board definition file
    # Bus 1 (pins 5, 3) is board SCL, SDA in the jetson definition file
    # Default is to Bus 1; We are using Bus 0, so we need to construct the busio first ...
    angle = int(sys.argv[1])

    # kit[0] is the bottom servo
    # kit[1] is the top servo
    print("Done initializing")
    
    kit.servo[14].angle=angle
    kit.servo[13].angle=angle
    os.system("echo 194 > /sys/class/gpio/export")
    os.system("echo out > /sys/class/gpio/gpio194/direction")
    try:
        while True:
            time.sleep(1)
            os.system("echo 1 > /sys/class/gpio/gpio194/value")
            time.sleep(1)
            os.system("echo 0 > /sys/class/gpio/gpio194/value")
            sweepTest()

    finally:
        pass

if __name__ == '__main__':
    main()