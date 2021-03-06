//g++ main.cpp dnn_opencv.cpp  -o test  `pkg-config opencv --cflags --libs`
#include <opencv2/highgui.hpp>
#include "dnn_opencv.h"

std::string absPath_weights = "/home/vn/darknet/";
std::string absPath_img = "/home/jav/wsl/images_videos/";
std::string classes = absPath_weights + "default/coco.names";
std::string weights = absPath_weights + "default/yolov3.weights";
std::string conf = absPath_weights + "default/yolov3.cfg";

dnn_opencv dnn = dnn_opencv();
int main(int argc, char const *argv[])
{
    dnn.load_model(conf, weights, classes);

    cv::VideoCapture cap("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)2592, height=(int)1458,format=(string)NV12, framerate=(fraction)30/1  ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv::CAP_GSTREAMER);
    cv::Mat frame;
    while (cap.isOpened())
    {
        cap >> frame;
        if (frame.empty())
        {
            break;
        }

        frame = dnn.inference(frame);
        dnn.getDetections(frame);
        //frame = dnn.drawDetection(frame);
        frame = dnn.drawFPS(frame); //Draw FPS

        cv::imshow("detections", frame);
        if (cv::waitKey(1) > 0)
        {
            break;
        }
    }
    cap.release();
    return 0;
}
