from scipy.misc import imread, imsave, toimage
import torch
from torch.autograd import Variable
from net_canny import Net
import cv2
import numpy as np

def canny(raw_img, use_cuda=False):
    img = torch.from_numpy(raw_img.transpose((2, 0, 1)))
    batch = torch.stack([img]).float()

    net = Net(threshold=3.0, use_cuda=use_cuda)
    if use_cuda:
        net.cuda()
    net.eval()

    data = Variable(batch)
    if use_cuda:
        data = Variable(batch).cuda()

    blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold = net(data)
    
    #imsave('gradient_magnitude.png',grad_mag.data.cpu().numpy()[0,0])
    blurred_PIL =toimage(blurred_img.data.cpu().numpy()[0,0])
    gradientPIL=toimage(grad_mag.data.cpu().numpy()[0,0])
    finalPIL=(thresholded.data.cpu().numpy()[0, 0] > 0.0).astype(float)
    earlyTresholdPIL= early_threshold.data.cpu().numpy()[0, 0]    
    

    #img1 = np.array(earlyTresholdPIL)
    #cv2.imshow('earlyTresholdPIL',img1)
    #img2 = np.array(blurred_PIL)
    #cv2.imshow('blurred_PIL',img2)
    #img3 = np.array(gradientPIL)
    #cv2.imshow('gradientPIL',img3)
    img4 = np.array(finalPIL)
    cv2.imshow('finalPIL',img4)
    #cv2.waitKey()
    #imsave('thin_edges.png', thresholded.data.cpu().numpy()[0, 0])
    #cv2.imshow('thin_edges.png',thresholded)
    
    #imsave('final.png', (thresholded.data.cpu().numpy()[0, 0] > 0.0).astype(float))
    #cv2.imshow('final.png',gradient_magnitude)
    
    #imsave('thresholded.png', early_threshold.data.cpu().numpy()[0, 0])
    #cv2.imshow('thresholded.png',gradient_magnitude)
    #cv2.waitKey()


if __name__ == '__main__':

    cap = cv2.VideoCapture("thermal1.mp4")

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Display the resulting frame
            cv2.imshow('Frame',frame)
            canny(frame, use_cuda=False)
            #canny(img, use_cuda=True)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else: 
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    #img = cv2.imread('img1.jpg') / 255.0

