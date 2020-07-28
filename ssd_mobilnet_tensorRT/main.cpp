#include "sampleSSD.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
extern const std::string gSampleName;
cv::Size original;
std::string weightsPath = "/home/tx2/wsl/weights/ssd/";

std::string get_tegra_pipeline(int width, int height, int fps)
{
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(width) + ", height=(int)" +
           std::to_string(height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(fps) +
           "/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

int main(int argc, char const *argv[])
{

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    SampleSSDParams params;
    params.dataDirs.push_back("");
    params.dataDirs.push_back("");
    params.dataDirs.push_back("");
    params.dataDirs.push_back("");
    params.prototxtFileName = weightsPath + "ssd.prototxt";
    params.weightsFileName = weightsPath + "VGG_VOC0712_SSD_300x300_iter_120000.caffemodel";
    params.inputTensorNames.push_back("data");
    params.batchSize = 1;
    params.outputTensorNames.push_back("detection_out");
    params.outputTensorNames.push_back("keep_count");
    //dlaCore  0 to n-1 DLA engine  cores for layer that support it
    params.dlaCore = -1;
    //int8 run  or fp16 run
    params.int8 = false;
    params.fp16 = false;

    params.outputClsSize = 21;
    params.keepTopK = 200; // Number of total bboxes to be kept per image after NMS step. It is same as detection_output_param.keep_top_k in prototxt file
    params.nbCalBatches = 500;
    params.visualThreshold = 0.6f;
    //params.calibrationBatches = "batches/batch_calibration";

    SampleSSD sample(params);

    gLogInfo << "Building and running a GPU inference engine for SSD" << std::endl;
    auto dnn_start = std::chrono::steady_clock::now();
    std::cout << "------------------------ BUILD " << std::endl;
    if (!sample.build())
    {
        std::cout << "build fail: " << std::endl;
        return gLogger.reportFail(sampleTest);
    }
    auto dnn_end = std::chrono::steady_clock::now();
    auto inference_fps = std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count();
    std::cout << "building time: " << inference_fps / 1000 << " seconds" << std::endl;
    cv::Mat image;

    // Options
    int WIDTH = 300;
    int HEIGHT = 300;
    int FPS = 60;

    // Define the gstream pipeline
    std::string pipeline = get_tegra_pipeline(WIDTH, HEIGHT, FPS);

    //cv::VideoCapture cap(0, cv::CAP_V4L);
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    while (cap.isOpened())
    {
        cap >> image;
        original = image.size();
        std::cout << "------------------------ inference " << std::endl;
        auto start = std::chrono::steady_clock::now();

        image = sample.infer(image);

        auto end = std::chrono::steady_clock::now();
        auto elapsed = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "total: " << elapsed << " FPS" << std::endl;
        cv::resize(image, image, original);
        cv::imshow("final", image);
        if (cv::waitKey(1) > 0)
        {
            break;
        }
    }
    std::cout << "------------------------ teardown " << std::endl;
    if (!sample.teardown())
    {
        std::cout << "teardown fail: " << std::endl;
        return gLogger.reportFail(sampleTest);
    }

    return gLogger.reportPass(sampleTest);
    return 0;
}
