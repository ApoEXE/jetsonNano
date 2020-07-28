//g++ main.cpp dnn_opencv.cpp  -o test  `pkg-config opencv --cflags --libs`
#include <opencv2/highgui.hpp>
#include "dnn_opencv.h"

std::string absPath_weights = "~/wsl/weights/yolov3-tiny/";
std::string classes = absPath_weights + "coco.names";
std::string weights = absPath_weights + "yolov3-tiny.weights";
std::string conf = absPath_weights + "yolov3-tiny.cfg";

dnn_opencv dnn = dnn_opencv();
int main(int argc, char const *argv[])
{
    printf("%s",argv[1]);
    dnn.load_model(conf, weights, classes);
    cv::VideoCapture cap;
    
    if(argc > 1)
        cap.open(argv[1], cv::CAP_FFMPEG);
    else
    {
        cap.open(0, cv::CAP_V4L);
    }
    
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
        frame = dnn.drawDetection(frame);
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
