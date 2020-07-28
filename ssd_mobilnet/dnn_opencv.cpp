#include "dnn_opencv.h"
dnn_opencv::dnn_opencv(/* args */)
{
}

dnn_opencv::~dnn_opencv()
{
}
void dnn_opencv::load_model(std::string lconf, std::string lweights, std::string lclasses)
{
    conf = lconf;
    weights = lweights;
    classes = lclasses;
    net = cv::dnn::readNetFromCaffe(conf, weights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    {
        std::ifstream class_file(classes);
        if (!class_file)
        {
            std::cerr << "failed to open " << classes << "\n";
        }
        class_names.assign(std::istream_iterator<std::string>(class_file), {});
    }

    output_names = net.getUnconnectedOutLayersNames();
    printf("passed loading model \n");
}

cv::Mat dnn_opencv::inference(cv::Mat frame)
{

    total_start = std::chrono::steady_clock::now();
    //blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    //cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(416, 416), cv::Scalar(), true, false, CV_32F);
    cv::dnn::blobFromImage(frame, blob, 0.007843, cv::Size(300, 300), cv::Scalar(127.5, 127.5, 127.5), false, false, CV_32F);
    net.setInput(blob);

    auto dnn_start = std::chrono::steady_clock::now();
    //net.forward(detections, output_names);
    cv::Mat output = net.forward();
    cv::Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());
    auto dnn_end = std::chrono::steady_clock::now();
    inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count();

    return detectionMat;
}

void dnn_opencv::getDetections(cv::Mat frame)
{
    boxes.clear();
    class_id.clear();
    scores.clear();
    for (auto &output : detections)
    {
        const auto num_boxes = output.rows;
        std::cout << num_boxes << std::endl;
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
}
cv::Mat dnn_opencv::drawDetection(cv::Mat detectionMat, cv::Mat frame)
{

    std::string classNames[] = {"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
    int confidenceThreshold = 0.7;
    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if (confidence > confidenceThreshold)
        {
            size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

            std::ostringstream ss;
            ss << confidence;
            std::string conf(ss.str());

            cv::Rect object((int)xLeftBottom, (int)yLeftBottom,
                            (int)(xRightTop - xLeftBottom),
                            (int)(yRightTop - yLeftBottom));

            rectangle(frame, object, cv::Scalar(0, 255, 0), 2);
            //std::cout << "----------------------------------" << std::endl;
            std::string label = std::string(classNames[objectClass]) + ": " + conf;
            //std::cout << "----------------------------------" << std::endl;
            int baseLine = 0;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            rectangle(frame, cv::Rect(cv::Point(xLeftBottom, yLeftBottom - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)),
                      cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(frame, label, cv::Point(xLeftBottom, yLeftBottom),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
    }
    return frame;
}

cv::Mat dnn_opencv::drawFPS(cv::Mat frame)
{
    std::chrono::_V2::steady_clock::time_point total_end = std::chrono::steady_clock::now();
    float total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    std::ostringstream stats_ss;
    stats_ss << std::fixed << std::setprecision(2);
    stats_ss << model << " Inference FPS: " << inference_fps << ", Total FPS: " << total_fps;
    stats = stats_ss.str();
    int baseline;
    auto stats_bg_sz = cv::getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
    cv::rectangle(frame, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255));
    return frame;
}

cv::Mat dnn_opencv::border(cv::Mat frame, cv::Mat mask)
{
    int lowerThr = 50;
    int higherThr = 200;
    cv::cuda::GpuMat frameGpu, gray_gpu, imageCanny_gpu, mask_gpu;
    frameGpu.upload(frame);
    mask_gpu.upload(mask);
    //cv::cuda::cvtColor(frameGpu,frameGpu,cv::COLOR_RGB2GRAY);
    //cv::cuda::cvtColor(frameGpu,frameGpu,cv::COLOR_GRAY2RGB);
    cv::cuda::cvtColor(frameGpu, gray_gpu, cv::COLOR_BGR2GRAY);
    cv::Ptr<cv::cuda::CannyEdgeDetector> canny_edg = cv::cuda::createCannyEdgeDetector(lowerThr, higherThr, 3, true);
    canny_edg->detect(gray_gpu, imageCanny_gpu);
    //DELETE borders from MASK
    if (!mask_gpu.empty())
    {
        //std::cout << "Entry 14" << std::endl;
        //std::cout << "Entry 18 imageCanny_gpu " << imageCanny_gpu.size() << "mask1_gpu " << mask1_gpu.size() << std::endl;
        if (mask_gpu.channels() == 3)
            cv::cuda::cvtColor(mask_gpu, mask_gpu, cv::COLOR_RGB2GRAY);
        //std::cout << "Entry 19 imageCanny_gpu " << imageCanny_gpu.channels() << " mask1_gpu " << mask1_gpu.channels() << std::endl;
        cv::cuda::GpuMat tempo; //mask with cross out section
        cv::cuda::resize(mask_gpu, tempo, imageCanny_gpu.size());

        cv::cuda::bitwise_and(imageCanny_gpu, tempo, imageCanny_gpu);
        //std::cout << "Entry 4" << std::endl;
        //std::cout << "Entry 15" << std::endl;
    }
    frameGpu.setTo(cv::Scalar(0, 255, 0), imageCanny_gpu); //draw green lines
    frameGpu.download(frame);
    return frame;
}