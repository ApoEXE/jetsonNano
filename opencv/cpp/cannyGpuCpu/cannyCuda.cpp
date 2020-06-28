#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <fstream>
#include <string>

//g++ main.cpp -o main `pkg-config opencv --cflags --libs`

int main(int argc, char *argv[])
{
    // open the default camera using default API
    // cap.open(0);
    // OR advance usage: select any API backend
    if (argc >= 2)
    {
        int deviceID = 2; // 0 = open default camera
        std::string videoPath = argv[1];
        printf(videoPath.c_str());
        int apiID = cv::CAP_V4L; // 0 = autodetect default API

        //CUDA

        cv::Mat frame2;
        //--- INITIALIZE VIDEOCAPTURE
        cv::VideoCapture cap2;
        //cap2.open(deviceID, apiID);
        cap2.open(videoPath);
   
        // check if we succeeded
        if (!cap2.isOpened())
        {
            std::cerr << "ERROR! Unable to open camera\n";
            return -1;
        }

        cv::cuda::GpuMat frameCuda;
        cv::cuda::GpuMat imageCanny_gpu;
        cv::cuda::GpuMat gray_gpu;
        cv::Mat downFrame;
        while (cap2.isOpened())
        {
            auto start = std::chrono::high_resolution_clock::now();
            /* code */
            cap2.read(frame2);
            frameCuda.upload(frame2); //paso de CPU a GPU
            //GPU
            cv::cuda::cvtColor(frameCuda, gray_gpu, cv::COLOR_BGR2GRAY);
            cv::Ptr<cv::cuda::CannyEdgeDetector> canny_edg = cv::cuda::createCannyEdgeDetector(20.0, 200.0, 5, true);
            canny_edg->detect(gray_gpu, imageCanny_gpu);
            frameCuda.setTo(cv::Scalar(0, 255, 0), imageCanny_gpu);

            frameCuda.download(downFrame);
            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_time = finish - start;
            std::cout << "GPU CUDA Elapsed Time: " << elapsed_time.count() * 1000 << " msecs"
                      << "\n"
                      << std::endl;
            cv::imshow("cuda", downFrame);
            if (cv::waitKey(5) >= 0)
                break;
        }
    }
    // cv::Mat downFrame; //safe edges in cpu
    //imageCanny_gpu.download(downFrame);
    //cv::Mat gpu_out;
    //ImageDev.download(gpu_out);
    //cv::imshow("GPU", gpu_out);

    return 0;
}