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

#ifdef _WIN32
#include <io.h>
#define access _access_s
#else
#include <unistd.h>
#endif

#include <iostream>
#include <unistd.h>
#include "conio.h"

#define top 119        //w
#define bottom 115     //s
#define left 97        //a
#define right 100      //d
#define scale_up 120   //x
#define scale_down 122 //z
#define esc 27         //esc
#define ratio 0.2      // w and h ratio
#define minSizePorce 0.1
#define l 108 // l minus lower
#define k 107 // l + lower
#define g 103 // h + higer
#define h 104 // h minus higer
int x = 0, y = 0;
uint16_t black_h = 0, black_w = 0;
uint16_t frame_h = 0, frame_w = 0;
uint16_t new_frame_h = 0, new_frame_w = 0;
//CANNY TRHESHOLDS
uint8_t lowerThr = 30;
uint8_t higherThr = lowerThr * 3;

std::string absPath_weights = "/home/jav/wsl/weights/";
std::string absPath_img = "/home/jav/wsl/images_videos/";
constexpr float confidence_threshold = 0.5;
constexpr float nms_threshold = 0.4;
constexpr int num_classes = 80;

// colors for bounding boxes
const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}};
const auto num_colors = sizeof(colors) / sizeof(colors[0]);
std::string classes = absPath_weights + "default/coco.names";
std::string weights = absPath_weights + "default/yolov4.weights";
std::string conf = absPath_weights + "default/yolov4.cfg";
//std::string videoPath = absPath_img+"busystreet.mp4";
//std::string videoPath = absPath_img+"thermalDriving.mp4";
std::string videoPath = absPath_img + "prueba.mp4";
std::string lookPath = absPath_img + "blackest.jpg";
#endif