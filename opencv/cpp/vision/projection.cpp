#include "projection.h"

projection::projection()
{
}

projection::~projection()
{
}

void projection::setValues(int cap_height, int cap_width, int lblack_w, int lblack_h)
{
    frame_h = cap_height; //ORIGINAL H
    frame_w = cap_width;  //ORIGINAL W
    //DEFAULT width and heoght of camera
    new_frame_h = frame_h; //START WITH ORIGINAL CAMERA H
    new_frame_w = frame_w; //START WITH ORIGINAL CAMERA W
    //LOAD A BLACK WALL PAPER AS BACKGROUND
    lblack_w = 1920; //I have preseted
    cv::Mat image(lblack_h, lblack_w, CV_8UC3, cv::Scalar(0, 0, 0));
    black = image;
    black_or = image;
    //SAVE BLACK PARAMETERS H W
    black_h = lblack_h; //BLACK H LIMIT FOR MOVING THE FRAME AROUND
    black_w = lblack_w; //BLACK W LIMIT FOR MOVING THE FRAME AROUND
    //printf("Black: width %d, height %d \n", black_w, black_h);
    //UPLOAD FRAME TO GPU
    blackGpu.upload(black);
    black_orGpu.upload(black_or);
    //DEFAULTS
    new_x = 0; //default position for videoWindow
    new_y = 0; //default position for videoWindow

    //CANNY TRHESHOLDS
    lowerThr = 50;
    higherThr = 200;

    //Modify DEFAULTS
    //Check if file exist //modify DEFAULTS load last x point videoFrame las y point, last width last height and thresholds
    if (FileExists("screen.conf"))
    {
        uploadConfig();
    }
    setPts("mask.conf");
}

cv::Mat projection::projectionFrame(cv::Mat lframe)
{
    cv::Mat downFrame;
    cv::cuda::GpuMat mask_lineCirPolly_gpu;
    //CUDA canny
    auto start = std::chrono::high_resolution_clock::now();
    resizeGpu.upload(lframe); //paso de CPU a GPU
    //DO PAINTINT MOUSE CALLSBACK AND SCALE UP
    cv::cuda::resize(resizeGpu, resizeGpu, cv::Size(new_frame_w, new_frame_h));
    if (!mask_lineCirPolly.empty())
    {
        //std::cout << "Entry 16" << std::endl;
        mask_lineCirPolly_gpu.upload(mask_lineCirPolly);
        cv::cuda::bitwise_or(resizeGpu, mask_lineCirPolly_gpu, resizeGpu);
    }
    //DRAW BACKGROUN AND MOVE FRAMECUDA TO NEW POSITION
    black_orGpu.copyTo(blackGpu);
    resizeGpu.copyTo(blackGpu(cv::Rect(new_x, new_y, resizeGpu.cols, resizeGpu.rows)));
    blackGpu.download(downFrame);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = finish - start;
    /*
    std::cout << "GPU CUDA Elapsed Time: " << elapsed_time.count() * 1000 << " msecs"
              << "\n"
              << std::endl;
              */
    return downFrame;
}

void projection::saveConfig()
{
    std::ofstream file;
    file.open("screen.conf", std::ios::trunc | std::ios::in);
    if (file.is_open())
    {

        file << "x_position," << new_x << "\n";
        file << "y_position," << new_y << "\n";
        file << "width," << new_frame_w << "\n";
        file << "hight," << new_frame_h << "\n";
        //printf("lower SAVECONFIG %d",lowerThr);
        file << "lowThr," << lowerThr << "\n";
        //printf("  HIGH SAVECONFIG %d \n",higherThr);
        file << "highThr," << higherThr;
        file.close();
    }
}

void projection::uploadConfig()
{

    std::fstream file;

    file.open("screen.conf", std::ios::in | std::ios::out);

    if (file.is_open())
    {

        std::string conf;
        while (!file.eof())
        {
            file >> conf;
            std::vector<std::string> token = split(conf, ",");
            //std::cout << token[0] << std::endl;
            //std::cout << token[1] << std::endl;
            if (strcmp(token[0].c_str(), "x_position") == 0)
            {
                new_x = std::stoi(token[1]);
            }
            if (strcmp(token[0].c_str(), "y_position") == 0)
            {
                new_y = std::stoi(token[1]);
            }
            if (strcmp(token[0].c_str(), "width") == 0)
            {
                new_frame_w = std::stoi(token[1]);
            }
            if (strcmp(token[0].c_str(), "hight") == 0)
            {
                new_frame_h = std::stoi(token[1]);
            }
            if (strcmp(token[0].c_str(), "lowThr") == 0)
            {
                lowerThr = std::stoi(token[1]);
            }
            if (strcmp(token[0].c_str(), "highThr") == 0)
            {
                higherThr = std::stoi(token[1]);
            }
        }
    }
    file.close();
}

std::vector<std::string> projection::split(const std::string &str, const std::string &delim)
{
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == std::string::npos)
            pos = str.length();
        std::string token = str.substr(prev, pos - prev);
        if (!token.empty())
            tokens.push_back(token);
        prev = pos + delim.length();
    } while (pos < str.length() && prev < str.length());
    return tokens;
}

void projection::saveMaskPts(std::vector<cv::Point> vpts)
{

    std::ofstream file;
    file.open("mask.conf", std::ios::trunc | std::ios::in);
    if (file.is_open())
    {
        for (auto &&i : vpts)
        {
            cv::Point pair = i;
            file << pair.x << "," << pair.y << "\n";
        }
        file.close();
    }
    //std::cout << "Entry 5" << std::endl;
    mask_1ch = returnMask(vpts, cv::Size(new_frame_w, new_frame_h));
    mask1_gpu.upload(mask_1ch);
}

cv::Mat projection::returnMask(std::vector<cv::Point> pts, cv::Size size)
{
    //RETURN POLYGON OF ZEROS AND REST OF 255
    cv::Mat mask1(size, CV_8UC1, cv::Scalar(20));
    std::vector<std::vector<cv::Point>> vpts;
    vpts.push_back(pts);

    //std::cout << "Entry 3" << std::endl;
    if (pts.size() > 2)
        cv::fillPoly(mask1, vpts, cv::Scalar(0, 0, 0), 8, 0);
    //std::cout << "Entry 4" << std::endl;

    return mask1;
}

cv::Mat projection::returnTransparency(float lalfa, cv::Mat src, cv::Mat mask)
{

    cv::addWeighted(mask, lalfa, src, 1 - lalfa, 0, src);
    return src;
}

bool projection::FileExists(const std::string &Filename)
{
    bool check = access(Filename.c_str(), 0) == 0;
    return check;
}

void projection::setMask(cv::Mat input)
{
    input.copyTo(mask_lineCirPolly);
}
cv::Size projection::getLastSize()
{

    return cv::Size(new_frame_w, new_frame_h);
}

void projection::setPts(std::string file)
{

    if (FileExists(file))
    {
        std::fstream file1;

        file1.open(file, std::ios::in | std::ios::out);

        if (file1.is_open())
        {
            std::string conf;
            while (!file1.eof())
            {
                file1 >> conf;
                std::vector<std::string> token = split(conf, ",");
                pts.push_back(cv::Point(std::stoi(token[0]), std::stoi(token[1])));
            }
        }

        file1.close();
        mask_1ch = returnMask(pts, cv::Size(new_frame_w, new_frame_h));
        mask1_gpu.upload(mask_1ch);
    }
    else
    {
    }
}
void projection::updatePts()
{

    float factorW = new_frame_w / frame_w;
    float factorH = new_frame_h / frame_h;
    std::vector<cv::Point> tmp_pts;
    for (auto &&i : pts)
    {
        tmp_pts.push_back(cv::Point((i.x + new_x), (i.y + new_y)));
    }
}

cv::Mat projection::Border(cv::Mat frame)
{
    cv::cuda::GpuMat frameGpu;
    frameGpu.upload(frame);
    cv::cuda::cvtColor(frameGpu, gray_gpu, cv::COLOR_BGR2GRAY);
    cv::Ptr<cv::cuda::CannyEdgeDetector> canny_edg = cv::cuda::createCannyEdgeDetector(lowerThr, higherThr, 3, true);
    canny_edg->detect(gray_gpu, imageCanny_gpu);
    //DELETE borders from MASK
    if (!mask1_gpu.empty())
    {
        //std::cout << "Entry 14" << std::endl;
        //std::cout << "Entry 18 imageCanny_gpu " << imageCanny_gpu.size() << "mask1_gpu " << mask1_gpu.size() << std::endl;
        if (mask1_gpu.channels() == 3)
            cv::cuda::cvtColor(mask1_gpu, mask1_gpu, cv::COLOR_RGB2GRAY);
        //std::cout << "Entry 19 imageCanny_gpu " << imageCanny_gpu.channels() << " mask1_gpu " << mask1_gpu.channels() << std::endl;
        cv::cuda::GpuMat tempo; //mask with cross out section
        cv::cuda::resize(mask1_gpu, tempo, imageCanny_gpu.size());

        cv::cuda::bitwise_and(imageCanny_gpu, tempo, imageCanny_gpu);
        //std::cout << "Entry 4" << std::endl;
        //std::cout << "Entry 15" << std::endl;
    }
    frameGpu.setTo(cv::Scalar(0, 255, 0), imageCanny_gpu); //draw green lines
    frameGpu.download(frame);
    return frame;
}