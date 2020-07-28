#ifndef DNN_OPENCV_H
#define DNN_OPENCV_H
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>


class dnn_opencv
{
private:
    /* data */
    cv::dnn::Net net; //load model here
    const float confidence_threshold = 0.5;
    const float nms_threshold = 0.4;
    const int num_classes = 3;

    std::string model = "yolov3";

    std::string classes;
    std::string weights;
    std::string conf;

    // detection
    cv::Mat blob;
    std::vector<cv::Mat> detections;
    std::vector<cv::Rect> boxes;
    std::vector<int> class_id;
    std::vector<float> scores;
    std::vector<cv::String> output_names;
    std::vector<std::string> class_names;
    float inference_fps;
    std::chrono::_V2::steady_clock::time_point total_start;
    std::vector<int> indices;
    
    // colors for bounding boxes
    const cv::Scalar colors[4] = {
        {0, 255, 255},
        {255, 255, 0},
        {0, 255, 0},//red
        {255, 0, 0}};//blue
    const int num_colors = sizeof(colors) / sizeof(colors[0]);

    //FPS
    std::__cxx11::string stats;

private:
    cv::Mat border(cv::Mat frame, cv::Mat mask);

public:
    dnn_opencv(/* args */);
    ~dnn_opencv();
    void load_model(std::string conf, std::string weights, std::string classes);
    cv::Mat inference(cv::Mat frame);
    cv::Mat drawDetection(cv::Mat detectionMat, cv::Mat frame);
    cv::Mat drawFPS(cv::Mat frame);
    void getDetections(cv::Mat frame);
};

#endif