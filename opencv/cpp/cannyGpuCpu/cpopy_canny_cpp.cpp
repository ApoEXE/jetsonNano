#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <fstream>

//g++ main.cpp -o main `pkg-config opencv --cflags --libs`

int main(int argc, char *argv[])
{
    cv::Mat ImageCanny, imageBlur, cpu_out, image_gray, imageMod;
    cv::namedWindow("cpu", cv::WINDOW_NORMAL);
    cv::namedWindow("canny", cv::WINDOW_NORMAL);

    std::cout << "Process started... \n"
              << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat original = cv::imread("cat.jpg");
    original.copyTo(imageMod);

    cv::cvtColor(original, image_gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(image_gray, imageBlur, cv::Size(5, 5), 1.5);
    cv::Canny(imageBlur, ImageCanny, 2, 200, 3, false);
    imageMod.setTo(cv::Scalar(0, 255, 0), ImageCanny);

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = finish - start;

    std::cout << "CPU Image Size: " << imageMod.rows * imageMod.cols << "  "
              << "Elapsed Time: " << elapsed_time.count() * 1000 << " msecs"
              << "\n"
              << std::endl;

    cv::imshow("cpu", original);
    cv::imshow("canny", imageMod);
    cv::waitKey(10000);


    /*
    //CUDA
    cv::cuda::GpuMat ImageDev;
    cv::cuda::GpuMat ImageGpu;
    ImageDev.upload(ImageHost);
    ImageGpu = ImageDev;
    cv::Ptr<cv::cuda::CannyEdgeDetector> canny_edg = cv::cuda::createCannyEdgeDetector(2.0, 100.0, 3, false);
    start = std::chrono::high_resolution_clock::now();
    canny_edg->detect(ImageDev, ImageGpu);
    finish = std::chrono::high_resolution_clock::now();
    elapsed_time = finish - start;

    std::cout << "GPU Image Size: " << ImageHost.rows * ImageHost.cols << "  "
              << "Elapsed Time: " << elapsed_time.count() * 1000 << " msecs"
              << "\n"
              << std::endl;

    std::cout << "Process ended... \n"
              << std::endl;
*/
    //cv::Mat gpu_out;
    //ImageDev.download(gpu_out);
    //cv::imshow("GPU", gpu_out);

    return 0;
}