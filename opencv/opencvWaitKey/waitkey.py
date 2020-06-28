import cv2
import threading
import sys
import select
import tty
import termios
import os
import numpy as np
import time
# Windows
if os.name == 'nt':
    import msvcrt

# Posix (Linux, OS X)
else:
    import sys
    import termios
    import atexit
    from select import select
img_size = 400
run_video = True
#background = np.zeros((600,800,3))
x_offset=y_offset=50
black_w = 1280
black_h = 960
img_w = 0
img_h = 0
#--font displays
fontScale = 0.5# Would work best for almost square images
thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX

class KBHit:

    def __init__(self):
        '''Creates a KBHit object that you can call to do various keyboard things.
        '''

        if os.name == 'nt':
            pass

        else:

            # Save the terminal settings
            self.fd = sys.stdin.fileno()
            self.new_term = termios.tcgetattr(self.fd)
            self.old_term = termios.tcgetattr(self.fd)

            # New terminal setting unbuffered
            self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)

            # Support normal-terminal reset at exit
            atexit.register(self.set_normal_term)


    def set_normal_term(self):
        ''' Resets to normal terminal.  On Windows this is a no-op.
        '''

        if os.name == 'nt':
            pass

        else:
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)


    def getch(self):
        ''' Returns a keyboard character after kbhit() has been called.
            Should not be called in the same program as getarrow().
        '''

        s = ''

        if os.name == 'nt':
            return msvcrt.getch().decode('utf-8')

        else:
            return sys.stdin.read(1)


    def getarrow(self):
        ''' Returns an arrow-key code after kbhit() has been called. Codes are
        0 : up
        1 : right
        2 : down
        3 : left
        Should not be called in the same program as getch().
        '''

        if os.name == 'nt':
            msvcrt.getch() # skip 0xE0
            c = msvcrt.getch()
            vals = [72, 77, 80, 75]

        else:
            c = sys.stdin.read(3)[2]
            vals = [65, 67, 66, 68]

        return vals.index(ord(c.decode('utf-8')))


    def kbhit(self):
        ''' Returns True if keyboard character was hit, False otherwise.
        '''
        if os.name == 'nt':
            return msvcrt.kbhit()

        else:
            dr,dw,de = select([sys.stdin], [], [], 0)
            return dr != []

def live_video(camera_port=0):
        global x_offset, y_offset, img_h, img_w
        video_capture = cv2.VideoCapture(camera_port,cv2.CAP_V4L)
        black = cv2.imread("blackest.jpg")
        black_original = cv2.imread("blackest.jpg")
        #print(black.shape)
        #frame = cv2.imread("img.png")
#frames per second calculation
        fps = 0
        end = 0.0
        frames = 0
        while run_video:
            start = time.time()
            # Capture frame-by-frame
            ret, frame= video_capture.read()

            img = preResize(frame)
            img_w = img.shape[1]
            img_h = img.shape[0]
            np.copyto(black,black_original)
            black[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
            cv2.putText(black,str(frames)+"FPS", (black.shape[1]-50,black.shape[0]-50),font , fontScale, (0,0,255),thickness)
            cv2.imshow("background",black)
            #cv2.imshow("black_ori",black_original)
            diff = time.time() - start
            end += diff
            
            fps +=1
            if(end >= 1.0):
                end=0
                frames = fps
                fps = 0
                #print(frames)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # When everything is done, release the capture
        #video_capture.release()
        cv2.destroyAllWindows()

def preResize(img):
#resize
    #print("preprocessCV")
    #print("W",img.shape[1],"H",img.shape[0])
    ratio = min(img_size/img.shape[1], img_size/img.shape[0])
    #print("ratio",ratio,"img_size/W",img_size/img.shape[1],"img_size/H",img_size/img.shape[0])
    imw = round(img.shape[1] * ratio)
    imh = round(img.shape[0] * ratio)
    #print("newW",imw,"newH",imh)
    dim = (imw, imh)
    img = cv2.resize(img, dim)
    return img

def prePadding(img):
      #padding
    top = 0
    bottom = 0
    left = 0
    right = 0
    if imw > imh:
        size = (imw-imh)/2
        pad_size = int(size)
        top = pad_size
        bottom = pad_size
        #print("top",top,"bottom",bottom)
    else :
        size = (imh-imw)/2
        pad_size = int(size)
        left = pad_size
        right = pad_size
        #print("left",left,"right",right)
    color = (128,128,128)
    data = cv2.copyMakeBorder( img, top, bottom,left,right, cv2.BORDER_CONSTANT,value=color)
    #print("nW",data.shape[1],"nH",data.shape[0])

    return data

def worker():
    global x_offset, y_offset, black_w, black_h, img_w,img_h
    key = ''
    kb = KBHit()

    print('Hit any key, or ESC to exit')

    while True:

        if kb.kbhit():
            key = kb.getch()
            print(key)
            if ord(key) == 27: # ESC
                break
            elif key == 'w':
                print("up")
                if(y_offset > 0):
                    y_offset -=1
            elif key == 's':
                print("down")
                if((y_offset+img_h) < black_h):
                    y_offset +=1
            elif key == 'a':
                print("left")
                if(x_offset > 0):
                    x_offset -=1
            elif key == 'd':
                print("right")
                if((x_offset+img_w) < black_w):
                    x_offset +=1
            print("x",x_offset,"y",y_offset)

    kb.set_normal_term()
    return


# Test    
if __name__ == "__main__":
    
    t = threading.Thread(target=worker)
    t.start()
    live_video(0)

    