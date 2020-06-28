// For more information and tips to improve inference FPS, visit https://github.com/opencv/opencv/pull/14827#issuecomment-568156546
//g++ yolov3_opencv_dnn_cuda.cpp -o test -pthread -lncurses `pkg-config opencv --cflags --libs`

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
constexpr int num_classes = 3;

// colors for bounding boxes
const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}};
const auto num_colors = sizeof(colors) / sizeof(colors[0]);

bool FileExists(const std::string &Filename)
{
    bool check = access(Filename.c_str(), 0) == 0;
    return check;
}

void *getKey(void *t_id)
{
    while (true)
    {
        if (kbhit() != 0)
        {
            int key = getch();
            //std::cout << key << std::endl;
            switch (key)
            {
            case top:
                std::cout << "TOP" << std::endl;
                if (y > 0)
                    y--;
                break;
            case bottom:
                std::cout << "BOTTOM" << std::endl;
                if ((y + new_frame_h) < black_h)
                    y++;
                break;
            case left:
                std::cout << "LEFT" << std::endl;
                if (x > 0)
                    x--;
                break;
            case right:
                std::cout << "RIGHT" << std::endl;
                if ((x + new_frame_w) < black_w)
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
int main(int argc, char *argv[])
{
    bool _default = true;
    std::string classes = absPath_weights + "default/coco.names";
    std::string weights = absPath_weights + "default/yolov3.weights";
    std::string conf = absPath_weights + "default/yolov3.cfg";
    //std::string videoPath = absPath_img+"busystreet.mp4";
    //std::string videoPath = absPath_img+"thermalDriving.mp4";
    std::string videoPath = absPath_img + "video1.mp4";
    std::string lookPath = absPath_img + "blackest.jpg";
    cv::Mat black = cv::imread(lookPath);
    cv::Mat black_or = cv::imread(lookPath);
    black_h = black.rows;
    black_w = black.cols;

    cv::cuda::GpuMat frameGpu;
    cv::cuda::GpuMat resizeGpu;
    cv::cuda::GpuMat blackGpu;
    cv::cuda::GpuMat black_orGpu;
    blackGpu.upload(black);
    black_orGpu.upload(black_or);
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
    
    if (_default)
    {
        classes = absPath_weights + "default/coco.names";
        weights = absPath_weights + "default/yolov3.weights";
        conf = absPath_weights + "default/yolov3.cfg";
    }
    else
    {
        conf = absPath_weights + "custom/yolov3-obj.cfg";

        weights = absPath_weights + "custom//yolov3-obj_last.weights";
        //weights = "custom/thermal8.weights";

        classes = absPath_weights + "custom/obj.names";
        std::cout<< "weights " << weights << " conf " << conf << " video path " << videoPath << " classes "<< classes<< "\n";
    }

    cv::cuda::GpuMat frameCuda;
    cv::cuda::GpuMat imageCanny_gpu;
    cv::cuda::GpuMat gray_gpu;
    cv::Mat downFrame;
    std::string model = "yolov3";
    std::vector<std::string> class_names;
    {
        std::ifstream class_file(classes);
        if (!class_file)
        {
            std::cerr << "failed to open " << classes << "\n";
            return 0;
        }
        class_names.assign(std::istream_iterator<std::string>(class_file), {});
    }
    if (!FileExists(weights) || !FileExists(conf) || !FileExists(videoPath))
    {
        std::cerr << "check which one is missing: " << weights << " " << conf << " " << videoPath << "\n";
        return 0;
    }
    

    //VIDEO CAPTURE
    cv::VideoCapture source;
    int input = std::stoi(argv[1]);
    if (argc >= 2 && input==2)
    {
        printf("Thermal ");
        source.open(2,cv::CAP_V4L);
    }
    if (argc >= 2 && input==0)
    {
        printf("webcam ");
        source.open(0,cv::CAP_V4L);
    }
    if (argc >= 3 && input==1)
    {
        printf("video Path ");
        source.open(argv[2],cv::CAP_FFMPEG);
    }
    

    //cv::VideoCapture source(videoPath);
    frame_h = source.get(cv::CAP_PROP_FRAME_HEIGHT);
    frame_w = source.get(cv::CAP_PROP_FRAME_WIDTH);
    new_frame_h = frame_h;
    new_frame_w = frame_w;

    //LOAD NET MODEL
    auto net = cv::dnn::readNetFromDarknet(conf, weights);

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    printf("passed 1\n");
    auto output_names = net.getUnconnectedOutLayersNames();

    cv::Mat frame, blob;
    std::vector<cv::Mat> detections;

    while (source.isOpened() > 0)
    {
        source >> frame;
        if (frame.empty())
        {
            cv::waitKey();
            break;
        }

        auto total_start = std::chrono::steady_clock::now();
        cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(416, 416), cv::Scalar(), true, false, CV_32F);
        net.setInput(blob);

        auto dnn_start = std::chrono::steady_clock::now();
        net.forward(detections, output_names);
        auto dnn_end = std::chrono::steady_clock::now();

        std::vector<cv::Rect> boxes;
        std::vector<int> class_id;
        std::vector<float> scores;

        for (auto &output : detections)
        {
            const auto num_boxes = output.rows;
            for (size_t i = 0; i < num_boxes; i++)
            {
                auto itr = std::max_element(output.ptr<float>(i, 5), output.ptr<float>(i, 5 + num_classes));
                auto confidence = *itr;
                auto classid = itr - output.ptr<float>(i, 5);
                if (confidence >= confidence_threshold)
                {
                    auto x = output.at<float>(i, 0) * frame.cols;
                    auto y = output.at<float>(i, 1) * frame.rows;
                    auto width = output.at<float>(i, 2) * frame.cols;
                    auto height = output.at<float>(i, 3) * frame.rows;
                    cv::Rect rect(x - width / 2, y - height / 2, width, height);

                    boxes.push_back(rect);
                    class_id.push_back(classid);
                    scores.push_back(confidence);
                }
            }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, 0.0, nms_threshold, indices);

        for (size_t i = 0; i < indices.size(); ++i)
        {
            const auto color = colors[i % num_colors];

            auto idx = indices[i];
            const auto &rect = boxes[idx];
            cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

            std::ostringstream label_ss;
            label_ss << class_names[class_id[idx]] << ": " << std::fixed << std::setprecision(2) << scores[idx];
            auto label = label_ss.str();

            int baseline;
            auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
            cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
            cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
        }

        auto total_end = std::chrono::steady_clock::now();

        float inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count();
        float total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
        std::ostringstream stats_ss;
        stats_ss << std::fixed << std::setprecision(2);
        stats_ss << model << " Inference FPS: " << inference_fps << ", Total FPS: " << total_fps;
        auto stats = stats_ss.str();

        int baseline;
        auto stats_bg_sz = cv::getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
        cv::rectangle(frame, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(frame, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));

        //CUDA canny

        auto start = std::chrono::high_resolution_clock::now();
        frameCuda.upload(frame); //paso de CPU a GPU
        //GPU
        cv::cuda::cvtColor(frameCuda, gray_gpu, cv::COLOR_BGR2GRAY);

        cv::Ptr<cv::cuda::CannyEdgeDetector> canny_edg = cv::cuda::createCannyEdgeDetector(lowerThr, higherThr, 3, true);
        canny_edg->detect(gray_gpu, imageCanny_gpu);
        frameCuda.setTo(cv::Scalar(0, 255, 0), imageCanny_gpu);

        //DO CONFIG OF SCREEN BEFORE SHOWING

        frameCuda.copyTo(resizeGpu);
        //std::cout << "CONFIG----------------------" << new_frame_w << " " << new_frame_h;
        cv::cuda::resize(resizeGpu, resizeGpu, cv::Size(new_frame_w, new_frame_h));
        black_orGpu.copyTo(blackGpu);
        resizeGpu.copyTo(blackGpu(cv::Rect(x, y, resizeGpu.cols, resizeGpu.rows)));
        blackGpu.download(downFrame);

        //frameCuda.download(downFrame);
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = finish - start;
        std::cout << "GPU CUDA Elapsed Time: " << elapsed_time.count() * 1000 << " msecs"
                  << "\n"
                  << std::endl;
        frame = downFrame;

        cv::namedWindow("output");
        cv::imshow("output", frame);
        cv::waitKey(1);
    }

    return 0;
}