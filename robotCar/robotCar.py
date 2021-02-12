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

from adafruit_servokit import ServoKit
#import RPi.GPIO as GPIO
import board
import busio
import time
import sys
import os
import cv2
#SYSFS
IN1_A = 50
IN2_A = 79
IN3_B = 194
IN4_B = 232
gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)1280, height=(int)720, '
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=2 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                       'videoconvert ! appsink').format(1920, 1080)
'''                       
print("Initializing Servos")
i2c_bus0=(busio.I2C(board.SCL_1, board.SDA_1))
print("Initializing ServoKit")
cam = ServoKit(channels=16, i2c=i2c_bus0)
print("Done initializing")
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

def setupGPIO():
    from os import walk
    path="/sys/class/gpio/"
    f = os.listdir(path)
    print(f)
    try:
        if (f.index("gpio"+str(IN1_A))):
            os.system("echo out > /sys/class/gpio/gpio"+str(IN1_A)+"/direction")
            pass
    except:
        os.system("echo "+str(IN1_A)+" > /sys/class/gpio/export")

    try:
        if (f.index("gpio"+str(IN2_A))):
            os.system("echo out > /sys/class/gpio/gpio"+str(IN2_A)+"/direction")
            pass
    except:
        os.system("echo "+str(IN2_A)+" > /sys/class/gpio/export")

    try:
        if (f.index("gpio"+str(IN3_B))):
            os.system("echo out > /sys/class/gpio/gpio"+str(IN3_B)+"/direction")
            pass
    except:
        os.system("echo "+str(IN3_B)+" > /sys/class/gpio/export")

    try:
        if (f.index("gpio"+str(IN4_B))):
            os.system("echo out > /sys/class/gpio/gpio"+str(IN4_B)+"/direction")
            pass
    except:
        os.system("echo "+str(IN4_B)+" > /sys/class/gpio/export")

#check dirrections
    


      
    
def left_forward():
    os.system("echo 0 > /sys/class/gpio/gpio"+str(IN3_B)+"/value")
    os.system("echo 1 > /sys/class/gpio/gpio"+str(IN4_B)+"/value")

def left_backward():
    os.system("echo 1 > /sys/class/gpio/gpio"+str(IN3_B)+"/value")
    os.system("echo 0 > /sys/class/gpio/gpio"+str(IN4_B)+"/value")

def right_backward():
    os.system("echo 0 > /sys/class/gpio/gpio"+str(IN1_A)+"/value")
    os.system("echo 1 > /sys/class/gpio/gpio"+str(IN2_A)+"/value")

def right_forward():
    os.system("echo 1 > /sys/class/gpio/gpio"+str(IN1_A)+"/value")
    os.system("echo 0 > /sys/class/gpio/gpio"+str(IN2_A)+"/value")

def stop():
    os.system("echo 0 > /sys/class/gpio/gpio"+str(IN1_A)+"/value")
    os.system("echo 0 > /sys/class/gpio/gpio"+str(IN2_A)+"/value")
    os.system("echo 0 > /sys/class/gpio/gpio"+str(IN3_B)+"/value")
    os.system("echo 0 > /sys/class/gpio/gpio"+str(IN4_B)+"/value")

def forward():
    print("fordware")
    right_forward()
    left_forward()

def backward():
    print("backward")
    right_backward()
    left_backward()

def to_left():
    print("to left")
    right_forward()
    left_backward()

def to_right():
    print("to right")
    right_backward()
    left_forward()

def testing_dc_motor():
    
    time.sleep(3)
    forward()
    time.sleep(3)
    backward()
    time.sleep(3)
    to_left()
    time.sleep(3)
    to_right()
    time.sleep(3)
    print("stop")
    stop()
    print("Done moving")

def main():
    # On the Jetson Nano
    # Bus 0 (pins 28,27) is board SCL_1, SDA_1 in the jetson board definition file
    # Bus 1 (pins 5, 3) is board SCL, SDA in the jetson definition file
    # Default is to Bus 1; We are using Bus 0, so we need to construct the busio first ...
    angle = int(sys.argv[1])


    #cam.servo[15].angle=angle
    print("Video Capture")
    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    print("cap objet")
    while(cap.isOpened()):
        _, img = cap.read()
        dim = (720, 480)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("camera", resized)
        key = cv2.waitKey(10)
        if key == 27: # ESC key: quit program
            break
        elif key == ord('W') or key == ord('w'): # toggle help message
            forward()
        elif key == ord('S') or key == ord('s'): # toggle help message
            backward()
        elif key == ord('D') or key == ord('d'): # toggle help message
            to_right()
        elif key == ord('A') or key == ord('a'): # toggle help message
            to_left()
        else:
            stop()
        key=0
    print("cam no opened")
    cap.release()
    cv2.destroyAllWindows()
    #testing_dc_motor()

    #sweepTest()

    #try:
        #while True:
            #sweepTest()

    #finally:
        #pass
    

from pynput.keyboard import Listener as KeyboardListener
def on_press(key):
    print("pressed: "+str(key))
    #pynput.keyboard._xorg.KeyCode
    if(str(key)=="'w'"):
       forward()
    if(str(key)=="'s'"):
        backward()
    if(str(key)=="'d'"):
        to_right()
    if(str(key)=="'a'"):
        to_left()

def on_release(key):
    print('{0} released'.format(key))
    stop()

if __name__ == '__main__':
    setupGPIO()
    os.system("echo out > /sys/class/gpio/gpio"+str(IN4_B)+"/direction")
       
    os.system("echo out > /sys/class/gpio/gpio"+str(IN3_B)+"/direction")
      
    os.system("echo out > /sys/class/gpio/gpio"+str(IN1_A)+"/direction")
       
    os.system("echo out > /sys/class/gpio/gpio"+str(IN2_A)+"/direction")

    os.system("export DISPLAY=:0")
    stop()
    # Setup the listener threads
    #keyboard_listener = KeyboardListener(on_press=on_press, on_release=on_release) 
    #keyboard_listener.start()
    #keyboard_listener.join() 
    
    main()
       