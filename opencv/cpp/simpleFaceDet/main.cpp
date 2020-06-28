//g++ main.cpp  -o test `pkg-config opencv --cflags --libs`
#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const float confidenceThreshold = 0.4;
const cv::Scalar meanVal(104.0, 177.0, 123.0);
float confidence;
#define CAFFE

const std::string caffeConfigFile = "deploy.prototxt.txt";
const std::string caffeWeightFile = "res10_300x300_ssd_iter_140000.caffemodel";
cv::Mat detectFaceOpenCVDNN(cv::dnn::dnn4_v20200310::Net net, cv::Mat lframeOpenCVDNN);


int main(int argc, const char **argv)
{
    std::string vid = "cara.avi";
    cv::dnn::dnn4_v20200310::Net net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    cv::VideoCapture source;

    source.open(vid, cv::CAP_FFMPEG);

    cv::Mat frame, cpyFrame;

    double tt_opencvDNN = 0;
    double fpsOpencvDNN = 0;

    int frame_width = source.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = source.get(cv::CAP_PROP_FRAME_HEIGHT);
 
    while (source.isOpened())
    {
        source >> frame;
        if (frame.empty())
            break;
        frame.copyTo(cpyFrame);
        double tim = cv::getTickCount();
        frame = detectFaceOpenCVDNN(net, frame);
        tt_opencvDNN = ((double)cv::getTickCount() - tim) / cv::getTickFrequency();
        fpsOpencvDNN = 1 / tt_opencvDNN;
        putText(frame, cv::format("FPS = %.2f Confidence: %0.2f", fpsOpencvDNN, confidence), cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        cv::imshow("OpenCV - DNN Face Detection", frame);

        int k = cv::waitKey(1) & 0xFF;
        if (k == 113)
        {
            std::cout << "Q" << k << std::endl;
            break;
        }
    }
    source.release();

    return 0;
}





cv::Mat detectFaceOpenCVDNN(cv::dnn::dnn4_v20200310::Net net, cv::Mat lframeOpenCVDNN)
{
    cv::Mat frameOpenCVDNN;
    lframeOpenCVDNN.copyTo(frameOpenCVDNN);
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
    //std::cout << "detectionMat.at<float>(0,0)" << detectionMat.at<float>(1, 0) << std::endl;

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
    return frameOpenCVDNN;
}