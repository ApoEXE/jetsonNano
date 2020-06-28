from models import *
from utils import *
from sort import *

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import cv2
import os, sys, time, datetime, random
import sys

#-------------------------------config neural network 
# load weights and set defaults
config_path='yolov3.cfg'
#weights_path='thermal8.weights'
weights_path='video8.weights'
class_path='obj.names'
#config_path='config/yolov3.cfg'
#weights_path='config/yolov3.weights'
#class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4

# load model and put into eval mode
model = Darknet(config_path, img_size=img_size)
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)
model.cuda()
model.load_weights(weights_path)

model.eval()

classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor

#tracker
mot_tracker = Sort()

#-------------------------------config neural network 

def detect_image(img):
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(img.to(device))
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]

def opencvToPytorch(img):
    #print("opencvToPytorch")
    #print("1#",img.shape)
    data = np.array(img, np.float32)
    #from [H][W][channel] to [channel][W][H]
    data = data.transpose((2, 0, 1))
    #print(type(data))
    data = data/255.0
    #print(type(data))
    #print("2#",data.shape)
    #data = data[0, :, :]
    #print(type(data))
    #print("3#",data.shape)
    # convert image to Tensor from [channel][W][H] to [1][channel][W][H]
    data = torch.from_numpy(data).float().unsqueeze_(0)
    data=Variable(data)
    return data

def preprocessCV(img):
    #resize
    print("preprocessCV")
    print("W",img.shape[1],"H",img.shape[0])
    ratio = min(img_size/img.shape[1], img_size/img.shape[0])
    print("ratio",ratio,"img_size/W",img_size/img.shape[1],"img_size/H",img_size/img.shape[0])
    imw = round(img.shape[1] * ratio)
    imh = round(img.shape[0] * ratio)
    #print("newW",imw,"newH",imh)
    dim = (imw, imh)
    img = cv2.resize(img, dim)

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

def drawDetecBox(img,detections):
    colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]
    
    #print("img.shape",img.shape)
    realWidth = img.shape[1]
    realHeight = img.shape[0]
    #new NN Image and realImage
    
    #print("realWidth",realWidth,"realHeight",realHeight)
    #transform to new scales
    scale = img_size / max(img.shape)
    pad_x = int(max((realHeight - realWidth), 0) * scale)
    pad_y = int(max(realWidth - realHeight, 0) * scale)
    #pixel values before padding but after resized
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    resized_h = unpad_h
    resized_w = unpad_w
   
    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu())
        unique_labels = detections[:, -1].cpu().unique()
        #print("found",len(unique_labels))
        # browse detections and draw bounding boxes
        #print("detections found",len(detections))
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            #print("x1",int(x1.item()),"y1",int(y1.item()),"x2",int(x2.item()),"y2",int(y2.item()),"obj_id",obj_id.item(),"cls_pred",cls_pred.item())
            #padSize added to the resized image Top and bottom left and right
            padSizeY = pad_y // 2
            padSizeX = pad_x // 2

            #padded point
            Y1P = y1
            X1P = x1
            X2P = x2
            Y2P = y2
            #finding "p1(x1R,y2R) resized value", padded Y1P = Y1R + padSizeY and padded X1P = X1R + padSizeX
            Y1R = (Y1P - padSizeY)
            X1R = (X1P - padSizeX)
            Y2R = (Y2P - padSizeY)
            X2R = (X2P - padSizeX)
            #ratio for transforming from resized to real representation
            ratioY1 = Y1R / resized_h
            ratioX1 = X1R / resized_w
            ratioY2 = Y2R / resized_h
            ratioX2 = X2R / resized_w
            #print("ratioY1R",ratioY1,"ratioX1R",ratioX1)
            #transform the points to real points
            y1 =  ratioY1 * realHeight
            x1 =  ratioX1 * realWidth
            y2 =  ratioY2 * realHeight
            x2 =  ratioX2 * realWidth
            y1 =  int(y1)
            x1 =  int(x1)
            y2 =  int(y2)
            x2 =  int(x2)
            color = colors[int(obj_id) % len(colors)]
            cls = classes[int(cls_pred)]
            cv2.rectangle(img,(x1,y1),(x2,y2),color,4)
            cv2.putText(img,cls+"-"+str(int(obj_id)), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),3)
    return img


def displayScaled(img,scale):
    processH = img.shape[0]
    processW = img.shape[1]
    #print("W",processW,"H",processH)
    imw = round(processW * scale)
    imh = round(processH * scale)
    #print("newW",imw,"newH",imh)
    dim = (imw, imh)
    img = cv2.resize(img, dim)
    return img



if __name__ == '__main__':
    videopath = str(sys.argv[1])
    cap = cv2.VideoCapture(videopath)
    width = int(cap.get(3))
    height = int(cap.get(4))
    print("width",width,"height",height)
    print(videopath)
    
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    #out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (width,height))
    #out = cv2.VideoWriter('output.avi',-1, 20.0, (width,height))
    totaltime = 0
    frames = 0
    fps = 0
    while (cap.isOpened()):
        starttime = time.time()
        ret, image = cap.read()
        if not ret:
            break
        cv2.imshow("imgage",image)
        frames += 1
        #prepare opencv img to tensor
        resizedPadded = preprocessCV(image)
        tensorImg = opencvToPytorch(resizedPadded)

        prev_time = time.time()
        
        #run inside Neural Network get x,y detection and classification
        detections = detect_image(tensorImg)

        #time taken by the Neural Network
        inference_time = datetime.timedelta(seconds=time.time() - prev_time)
        #print ('Inference Time: %s' % (inference_time))

        #draw boxes and classification
        imgDetected = drawDetecBox(image,detections)

        scale = max((imgDetected.shape[1]/width),(imgDetected.shape[0]/height))
        #imgDetected= displayScaled(imgDetected,scale)
        #print("width",imgDetected.shape[1],"height",imgDetected.shape[0], "Realwidth",width,"RealHeight",height,"Scale:",scale)
        cv2.putText(imgDetected,str(fps)+"FPS", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
        #out.write(imgDetected)
        cv2.imshow("detection",imgDetected)
        cv2.waitKey(3000)
        totaltime += time.time()-starttime
        if(totaltime >= 1):
            print(frames, "FPS")
            fps = frames
            starttime = 0
            totaltime = 0
            frames = 0

cap.release()
out.release()
cv2.destroyAllWindows()

