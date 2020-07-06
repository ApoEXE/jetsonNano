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
    net = cv::dnn::readNetFromDarknet(conf, weights);
    //net = cv::dnn::readNet(conf, weights);
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

    cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(416, 416), cv::Scalar(), true, false, CV_32F);
    net.setInput(blob);

    auto dnn_start = std::chrono::steady_clock::now();
    net.forward(detections, output_names);
    auto dnn_end = std::chrono::steady_clock::now();
    inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count();

    return frame;
}

void dnn_opencv::getDetections(cv::Mat frame)
{
    boxes.clear();
    class_id.clear();
    scores.clear();
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
}
cv::Mat dnn_opencv::drawDetection(cv::Mat frame)
{

    cv::dnn::NMSBoxes(boxes, scores, 0.0, nms_threshold, indices);

    for (size_t i = 0; i < indices.size(); ++i)
    {
        cv::Scalar color;

        auto idx = indices[i];
        const auto &rect = boxes[idx];

        std::ostringstream label_ss;
        // label_ss << class_names[class_id[idx]] << ": " << std::fixed << std::setprecision(2) << scores[idx];
        label_ss << class_names[class_id[idx]];
        auto label = label_ss.str();

        color = cv::Scalar(0, 128, 255);

        //COORD
        int baseline;
        auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, 0.5, &baseline);
        int x_coor = rect.x;
        int y_coor = rect.y + baseline + 5 + rect.height;
        int p1coor_x = x_coor;
        int p1coor_y = rect.y + label_bg_sz.height + baseline + rect.height;
        int p2coor_x = rect.x + label_bg_sz.width;
        int p2coor_y = rect.y + rect.height;
        cv::Point text = cv::Point(x_coor, y_coor);
        cv::Point lbPoint1 = cv::Point(p1coor_x, p1coor_y);
        cv::Point lbPoint2 = cv::Point(p2coor_x, p2coor_y);
        cv::Mat mask(frame.size(), CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Mat mask1(frame.size(), CV_8UC3, cv::Scalar(0, 0, 0));
        cv::rectangle(mask1, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), cv::Scalar(255, 255, 255), cv::FILLED);
        cv::bitwise_or(frame, mask, mask);
        frame = border(frame,mask1);
        //cv::imshow("mask dnn",mask);
        cv::rectangle(mask, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), cv::Scalar(0, 0, 255), cv::FILLED);
        float lalfa = 0.3;
        cv::addWeighted(mask, lalfa, frame, 1 - lalfa, 0, frame);
        //BOX DETECTION
        cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 1);
        //BACKGROUIND FOR TEXT
        cv::rectangle(frame, lbPoint1, lbPoint2, color, cv::FILLED);
        cv::putText(frame, label.c_str(), text, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 255, 255));
        //cv::circle(frame, cv::Point(rect.x + label_bg_sz.width - 5, rect.y + rect.height), 2, cv::Scalar(0, 0, 255), -1, 8, 0); //creates a circle point
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
    cv::putText(frame, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));
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