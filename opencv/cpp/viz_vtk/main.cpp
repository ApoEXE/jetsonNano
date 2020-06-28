//g++ main.cpp -o test  `pkg-config opencv --cflags --libs`
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <iostream>
using namespace std;
using namespace cv;
std::string absPath_img = "/home/jav/wsl/images_videos/";
std::string videoPath = absPath_img + "video1.mp4";
void KeyboardViz3d(const viz::KeyboardEvent &w, void *t)
{
    viz::Viz3d *fen = (viz::Viz3d *)t;
    if (w.action)
        cout << "you pressed " << w.code << " = " << w.symbol << " in viz window " << fen->getWindowName() << "\n";
}

int main(int argc, char **argv)
{

    VideoCapture cap(videoPath, CAP_FFMPEG);
    Mat img;
    //cvtColor(img, imgh, CV_BGR2GRAY);

    viz::Viz3d myWindow("Viz window");
    myWindow.registerKeyboardCallback(KeyboardViz3d, &myWindow);
    myWindow.setBackgroundColor(cv::viz::Color::black());
    while (!myWindow.wasStopped())
    {
        cap >> img;
        if(img.empty()){
            break;
        }
        myWindow.showImage(img, img.size());
        myWindow.spinOnce(1, true);
    }

    return 0;
}
