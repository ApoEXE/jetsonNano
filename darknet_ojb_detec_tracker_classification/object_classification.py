from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable


import cv2
import sys
#-------------------------------config neural network 



if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)

#weights_path='thermal8.weights'
weights_path='video8.weights'
class_path='obj.names'
config_path='thermal_yolov3.cfg'
#config_path='config/yolov3.cfg'
#weights_path='config/yolov3.weights'
#class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4
# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()

classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor

#--------------------------------
def detect_image(img):
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(img.to(device))
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]


def opencvToPytorch(img):
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
    print("W",img.shape[1],"H",img.shape[0])
    ratio = min(img_size/img.shape[1], img_size/img.shape[0])
    #print("ratio",ratio,"img_size/W",img_size/img.shape[1],"img_size/H",img_size/img.shape[0])
    imw = round(img.shape[1] * ratio)
    imh = round(img.shape[0] * ratio)
    print("newW",imw,"newH",imh)
    dim = (imw, imh)
    img = cv2.resize(img, dim)
    print("W2",img.shape[1],"H2",img.shape[0])
    cv2.imshow("resized",img)
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
        print("top",top,"bottom",bottom)
    else :
        size = (imh-imw)/2
        pad_size = int(size)
        left = pad_size
        right = pad_size
        #print("left",left,"right",right)
    color = (128,128,128)
    data = cv2.copyMakeBorder( img, top, bottom,left,right, cv2.BORDER_CONSTANT,value=color)
    #print("nW",data.shape[1],"nH",data.shape[0])
    cv2.imshow("padded", data)

    return data


def drawDetecBox(img,detections):
    #print("img.shape",img.shape)
    realWidth = img.shape[1]
    realHeight =img.shape[0]
    #print("realWidth",realWidth,"realHeight",realHeight)
    pad_x = max(realHeight - realWidth, 0) * (img_size / max(img.shape))
    pad_y = max(realWidth - realHeight, 0) * (img_size / max(img.shape))
    #pixel values before padding but after resized
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    resized_h = unpad_h
    resized_w = unpad_w
    
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        print("Classes found",len(unique_labels),"detections found",len(detections))
        # browse detections and draw bounding boxes
        print()
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            #padSize added to the resized image Top and bottom left and right
            '''
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
            '''
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),thickness=2)
            cv2.putText(img,classes[int(cls_pred)], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
    else:
        print("didnt find detecctions")       
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
    img_path = str(sys.argv[1])
    print(img_path)
    #img_path = "images/blueangels.jpg"
    #img_path = "images/Intersection-Counts.jpg"
    prev_time = time.time()
    image = cv2.imread(img_path)
    #cv2.imshow("original",image)
    resizedPadded = preprocessCV(image)
    tensorImg = opencvToPytorch(resizedPadded)
    detections = detect_image(tensorImg)
    inference_time = datetime.timedelta(seconds=time.time() - prev_time)
    print ('Inference Time: %s' % (inference_time))
    imgDetected = drawDetecBox(resizedPadded,detections)
    cv2.imshow("img_with_detec", imgDetected)
    #scale = 0.3
    #imgDetected= displayScaled(imgDetected,scale)
    #cv2.imshow("imagen",imgDetected)
    while(1):
        k = cv2.waitKey(33)
