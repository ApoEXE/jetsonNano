//g++ main.cpp dnn_opencv.cpp  -o test  `pkg-config opencv --cflags --libs`
#include <opencv2/highgui.hpp>
#include "dnn_opencv.h"

std::string get_tegra_pipeline(int width, int height, int fps)
{
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(width) + ", height=(int)" +
           std::to_string(height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(fps) +
           "/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

int main(int argc, char const *argv[])
{
    // Options
    int WIDTH = 1280;
    int HEIGHT = 720;
    int FPS = 60;
    std::string pipeline = get_tegra_pipeline(WIDTH, HEIGHT, FPS);
    dnn_opencv dnn = dnn_opencv();
    dnn.load_model("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel", "classes");

    // Options
    int WIDTH = 300;
    int HEIGHT = 300;
    int FPS = 60;

    // Define the gstream pipeline
    std::string pipeline = get_tegra_pipeline(WIDTH, HEIGHT, FPS);

    //cv::VideoCapture cap(0, cv::CAP_V4L);
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    //cv::VideoCapture cap = cv::VideoCapture(0, cv::CAP_V4L);
    cv::Mat frame;
    while (cap.isOpened())
    {
        cap >> frame;
        if (frame.empty())
        {
            break;
        }
        cv::resize(frame,frame,cv::Size(300,300));
        cv::Mat detectionMat = dnn.inference(frame);
       // dnn.getDetections(frame);
        frame = dnn.drawDetection(detectionMat,frame);
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
