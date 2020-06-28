#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <fstream>

//g++ main.cpp -o main `pkg-config opencv --cflags --libs`
#define SIZE 10

int main(int argc, char *argv[])
{
    cv::Mat ImageHost = cv::imread("1.jpg", cv::IMREAD_GRAYSCALE);

    cv::Mat ImageHostArr[SIZE];

    cv::cuda::GpuMat ImageDev;
    cv::cuda::GpuMat ImageDevArr[SIZE];

    ImageDev.upload(ImageHost);

    for (int n = 1; n < SIZE; n++)
        cv::resize(ImageHost, ImageHostArr[n], cv::Size(), 0.5 * n, 0.5 * n, cv::INTER_LINEAR);

    for (int n = 1; n < SIZE; n++)
        cv::cuda::resize(ImageDev, ImageDevArr[n], cv::Size(), 0.5 * n, 0.5 * n, cv::INTER_LINEAR);

    cv::Mat Detected_EdgesHost[SIZE];
    cv::cuda::GpuMat Detected_EdgesDev[SIZE];

    std::ofstream File1, File2;

    File1.open("canny_cpu.txt");
    File2.open("canny_gpu.txt");

    std::cout << "Process started... \n"
              << std::endl;
    for (int n = 1; n < SIZE; n++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        cv::Canny(ImageHostArr[n], Detected_EdgesHost[n], 2.0, 100.0, 3, false);
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = finish - start;
        File1 << "Image Size: " << ImageHostArr[n].rows * ImageHostArr[n].cols << "  "
              << "Elapsed Time: " << elapsed_time.count() * 1000 << " msecs"
              << "\n"
              << std::endl;
        cv::imshow("cpu",Detected_EdgesHost[n]);
        cv::waitKey(1000);
    }

    
        cv::Ptr<cv::cuda::CannyEdgeDetector> canny_edg = cv::cuda::createCannyEdgeDetector(2.0, 100.0, 3, false);   



        for (int n = 1; n < SIZE; n++) {
            auto start = std::chrono::high_resolution_clock::now();
            canny_edg->detect(ImageDevArr[n], Detected_EdgesDev[n]);
            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_time = finish - start;
            File2 << "Image Size: " << ImageDevArr[n].rows* ImageDevArr[n].cols << "  " << "Elapsed Time: " << elapsed_time.count() * 1000 << " msecs" << "\n" << std::endl;
        }
        



    std::cout << "Process ended... \n"
              << std::endl;
    return 0;
}