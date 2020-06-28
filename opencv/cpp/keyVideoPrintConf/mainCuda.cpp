//g++ mainCuda.cpp -o test -pthread -lncurses `pkg-config opencv --cflags --libs`

#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <pthread.h>

#include <iostream>
#include <unistd.h>
#include "conio.h"
#define top 119          //w
#define bottom 115       //s
#define left 97          //a
#define right 100        //d
#define scale_up 120     //x
#define scale_down 122   //z
#define esc 27           //esc
#define ratio 0.2        // w and h ratio
#define minSizePorce 0.1 //min % of W to decrease in size
#define l 108            // l minus lower
#define k 107            // l + lower
#define g 103            // h + higer
#define h 104            // h minus higer

std::string absPath_img = "/home/jav/wsl/images_videos/";
int x = 0, y = 0;
uint16_t black_h = 0, black_w = 0;
uint16_t frame_h = 0, frame_w = 0;
uint16_t new_frame_h = 0, new_frame_w = 0;

uint8_t lowerThr = 0;
uint8_t higherThr = 0;
void *getKey(void *t_id)
{
    while (true)
    {
        if (kbhit() != 0)
        {
            int key = getch();
            std::cout << key << std::endl;
            switch (key)
            {
            case top:
                std::cout << "TOP" << std::endl;
                if (y > 0)
                    y--;
                break;
            case bottom:
                std::cout << "BOTTOM" << std::endl;
                if ((y + frame_h) < black_h)
                    y++;
                break;
            case left:
                std::cout << "LEFT" << std::endl;
                if (x > 0)
                    x--;
                break;
            case right:
                std::cout << "RIGHT" << std::endl;
                if ((x + frame_w) < black_w)
                    x++;
                break;
            case scale_up:

                if (new_frame_w < black_w && ((new_frame_w + x) + (new_frame_w * ratio)) < black_w && new_frame_h < black_h && ((new_frame_h + y) + (new_frame_h * ratio)) < black_h)
                {
                    std::cout << "scale up" << std::endl;
                    new_frame_w += new_frame_w * ratio;
                    new_frame_h += new_frame_h * ratio;
                }

                break;
            case scale_down:

                if (new_frame_w > (minSizePorce * frame_w))
                {
                    std::cout << "scale down" << std::endl;
                    new_frame_w -= new_frame_w * ratio;
                    new_frame_h -= new_frame_h * ratio;
                }
                break;
            case l:

                if (lowerThr > 0)
                {
                    std::cout << "Low threshold Canny" << std::endl;
                    lowerThr -= 1;
                }

                break;
            case k:

                if (lowerThr < 255)
                {
                    std::cout << "Low threshold Canny" << std::endl;
                    lowerThr += 1;
                }
                break;
            case h:

                if (higherThr > 0)
                {
                    std::cout << "Higher threshold Canny" << std::endl;
                    higherThr -= 1;
                }

                break;
            case g:

                if (higherThr < 999)
                {
                    std::cout << "Higher threshold Canny" << std::endl;
                    higherThr += 1;
                }
                break;
            case esc:
                exit(0);
                break;

            default:
                std::cout << "KEY NOT MAPPED" << std::endl;
                break;
            }
        }
    }
}
int main()
{

    std::string lookPath = absPath_img + "blackest.jpg";
    cv::VideoCapture cap(0, cv::CAP_V4L);
    cv::Mat black = cv::imread(lookPath);
    cv::Mat black_or = cv::imread(lookPath);
    black_h = black.rows;
    black_w = black.cols;
    //Thread
    uint8_t id = 1;
    int rc = 0;
    pthread_t t;
    rc = pthread_create(&t, NULL, getKey, (void *)id);
    if (rc)
    {
        std::cout << "Error:unable to create thread," << rc << std::endl;
        exit(-1);
    }

    if (cap.isOpened() > 0)
    {
        frame_h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        frame_w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        new_frame_h = frame_h;
        new_frame_w = frame_w;
        cv::Mat frame;
        cv::Mat resize;
        std::cout << "CAM width " << frame_w << "CAM height " << frame_h << std::endl;

        cv::cuda::GpuMat frameGpu;
        cv::cuda::GpuMat resizeGpu;
        cv::cuda::GpuMat blackGpu;
        cv::cuda::GpuMat black_orGpu;
        blackGpu.upload(black);
        black_orGpu.upload(black_or);
        while (cap.isOpened())
        {

            cap.read(frame);
            frameGpu.upload(frame);
            frameGpu.copyTo(resizeGpu);
            cv::cuda::resize(resizeGpu, resizeGpu, cv::Size(new_frame_w, new_frame_h));
            black_orGpu.copyTo(blackGpu);
            resizeGpu.copyTo(blackGpu(cv::Rect(x, y, resizeGpu.cols, resizeGpu.rows)));
            blackGpu.download(black);
            cv::imshow("background", black);
            //cv::imshow("video", frame);
            if (cv::waitKey(5) >= 0)
                break;
        }
    }
    pthread_exit(NULL);
    return 0;
}