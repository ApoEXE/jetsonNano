//g++ main.cpp -o test -pthread -lncurses `pkg-config opencv --cflags --libs`

#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <future>
#include <thread>
#include <chrono>

#include <pthread.h>

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
std::string absPath_img = "/home/jav/wsl/images_videos/";
int x = 0, y = 0;
uint16_t black_h = 0, black_w = 0;
uint16_t frame_h = 0, frame_w = 0;
uint16_t new_frame_h = 0, new_frame_w = 0;
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
                
                if (new_frame_w < black_w && ((new_frame_w +x)+(new_frame_w*ratio)) < black_w && new_frame_h < black_h && ((new_frame_h +y)+(new_frame_h*ratio)) < black_h){
                    std::cout << "scale up" << std::endl;
                    new_frame_w += new_frame_w * ratio;
                    new_frame_h += new_frame_h * ratio;
                }

                break;
            case scale_down:
            
                if (new_frame_w > (minSizePorce * frame_w)){
                    std::cout << "scale down" << std::endl;
                    new_frame_w -= new_frame_w * ratio;
                    new_frame_h -= new_frame_h * ratio;
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
        while (cap.isOpened())
        {

            cap.read(frame);
            frame.copyTo(resize);

            cv::resize(resize, resize, cv::Size(new_frame_w, new_frame_h));
            black_or.copyTo(black);
            resize.copyTo(black(cv::Rect(x, y, resize.cols, resize.rows)));
            cv::imshow("background", black);
            //cv::imshow("video", frame);
            if (cv::waitKey(5) >= 0)
                break;
        }
    }
    pthread_exit(NULL);
    return 0;
}