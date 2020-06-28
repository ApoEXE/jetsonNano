//g++ main.cpp -o test `pkg-config opencv --cflags --libs`

#include "opencv2/opencv.hpp"
using namespace cv;

int main(int, char**)
{
    VideoCapture cap("/home/jav/Downloads/video1Images/video1.mp4"); // open the default camera
    if (!cap.isOpened())  // check if we succeeded
        return -1;

    Mat frame;
    int count = 1000;

    for (;;)
    {
        // Get frame
        cap >> frame; // get a new frame from camera

        // Update counter
        ++count;


        imshow("frame", frame);


        // Save foreground mask
        std::string name =std::to_string(count) + ".png";
        imwrite("/home/jav/Downloads/video1Images/" + name, frame);

        if (waitKey(1) >= 0) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}