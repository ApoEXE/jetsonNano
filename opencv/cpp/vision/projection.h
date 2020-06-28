#ifndef PROJECTION_H
#define PROJECTION_H

#include <iostream>
#include <unistd.h>
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
#include <fstream>

#define top 119        //w
#define bottom 115     //s
#define left 97        //a
#define right 100      //d
#define scale_up 120   //x
#define scale_down 122 //z
#define esc 27         //esc
#define ratio 0.2      // w and h ratio
#define minSizePorce 0.1
#define low_low 108 // l minus lower
#define low_high 107 //  + lower
#define high_high 103 // h + higer
#define high_low 104 // h minus higer
class projection
{
private:
    int cap_height;
    int cap_width;
    cv::Mat black;
    cv::Mat black_or;
    cv::Mat noInclude1;//mask to not include black portion

    cv::cuda::GpuMat resizeGpu;      //SAVE RESIZE FRAME BUT GPU
    cv::cuda::GpuMat blackGpu;       //SAVE RESIZE FRAME BUT GPU
    cv::cuda::GpuMat black_orGpu;    //SAVE RESIZE FRAME BUT GPU
    cv::cuda::GpuMat frameCuda;      //SAVE FRAME FRAME BUT GPU
    cv::cuda::GpuMat imageCanny_gpu; //SAVE CANNY LINES FRAME BUT GPU
    cv::cuda::GpuMat gray_gpu;       //SAVE GRAYSCALE FRAME BUT GPU


    std::vector<cv::Point> pts;
    cv::Mat mask_1ch;//here mask from vector is save, first mask
    cv::cuda::GpuMat mask1_gpu;       //mask to bitwise out of canny

    cv::Mat mask_lineCirPolly;//here where mask for circle line and poly customization takes place

    std::vector<std::string> split(const std::string& str, const std::string& delim);
    bool FileExists(const std::string &Filename);
    void setPts(std::string file);

public:
    uint16_t new_x,new_y;
    int frame_h, frame_w, new_frame_h, new_frame_w, black_h, black_w;
    //CANNY TRHESHOLDS
    uint16_t lowerThr;
    uint16_t higherThr;


    projection();
    ~projection();
    cv::Mat projectionFrame(cv::Mat frame);
    void setValues(int cap_height, int cap_width, int black_w, int black_h);
    void saveConfig();
    void uploadConfig();
    void saveMaskPts(std::vector<cv::Point> vpts);
    cv::Mat returnMask(std::vector<cv::Point> vpts, cv::Size size);
    cv::Mat returnTransparency(float alfa, cv::Mat src, cv::Mat mask);
    cv::Mat applyPolyMask(cv::Mat input);
    void setMask(cv::Mat);
    cv::Size getLastSize();//get the values of the final image size to be process
    void updatePts();
    cv::Mat Border(cv::Mat frame);


};

#endif