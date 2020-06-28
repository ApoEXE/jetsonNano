#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;
using namespace cv::dnn;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const float confidenceThreshold = 0.4;
const cv::Scalar meanVal(104.0, 177.0, 123.0);
float confidence;
#define CAFFE

const std::string caffeConfigFile = "deploy.prototxt.txt";
const std::string caffeWeightFile = "res10_300x300_ssd_iter_140000.caffemodel";

void detectFaceOpenCVDNN(Net net, Mat &frameOpenCVDNN)
{
    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;

    cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false, false);

    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
    //rows detection.size[2] 200
    //cols detection.size[3] 7
    //type CV_32F
    //(rows,0): 0
    //(rows,1): 1
    //(rows,2): confidence
    //(rows,3): x1
    //(rows,4): y1
    //(rows,5): x2
    //(rows,6): y2
    std::cout << "detectionMat.at<float>(0,0)" << detectionMat.at<float>(1,0) << std::endl;

    cv::imshow("decMat", detectionMat);
    for (int i = 0; i < detectionMat.rows; i++)
    {

        float lconfidence = detectionMat.at<float>(i, 2);
        //cout << confidence << endl;

        if (lconfidence > confidenceThreshold)
        {
            confidence = lconfidence;
            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

            cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2, 4);
        }
    }
}

int main(int argc, const char **argv)
{
    cv::dnn::dnn4_v20200310::Net net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    VideoCapture source;

    source.open(0, cv::CAP_V4L);

    Mat frame;

    double tt_opencvDNN = 0;
    double fpsOpencvDNN = 0;
    while (1)
    {
        source >> frame;
        if (frame.empty())
            break;
        double tim = cv::getTickCount();
        detectFaceOpenCVDNN(net, frame);
        tt_opencvDNN = ((double)cv::getTickCount() - tim) / cv::getTickFrequency();
        fpsOpencvDNN = 1 / tt_opencvDNN;
        putText(frame, format("FPS = %.2f Confidence: %0.2f", fpsOpencvDNN, confidence), Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
        imshow("OpenCV - DNN Face Detection", frame);
        int k = waitKey(1);
    }
}