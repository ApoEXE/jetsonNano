//g++ main.cpp dnn_opencv.cpp projection.cpp -o test -pthread -lncurses -lX11 `pkg-config opencv --cflags --libs`
//./test 1 ~/wsl/images_videos/video1.mp4
#include "dnn_opencv.h"
#include "projection.h"
#include <X11/Xlib.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <unistd.h>
#define DNN
#define PROJETION
#define KEYBOARD
#ifdef KEYBOARD
#include "conio.h"
#endif
#ifdef PROJETION
projection pro = projection();
#endif
#ifdef DNN
dnn_opencv dnn = dnn_opencv();
#endif
int screen_num;    //number of display
int screen_width;  //width of display
int screen_height; //height of display
Window root_window;
unsigned long white_pixel;
unsigned long black_pixel;

// MOUSE DATA
cv::Mat src, mask, final;
cv::Mat *p_src;
cv::Point point;
std::vector<cv::Point> pts;
int drag = 0;
int var = 0; //if there are more than one point create a line
int flag = 0;
int capture_height = 0;
int capture_width = 0;
bool rightDown = false;
bool clearMask = false;
bool rightUp = false;

std::string model = "yolov3";

std::string absPath_weights = "/home/jav/wsl/weights/";
std::string absPath_img = "/home/jav/wsl/images_videos/";
std::string classes = absPath_weights + "default/coco.names";
std::string weights = absPath_weights + "default/yolov3.weights";
std::string conf = absPath_weights + "default/yolov3.cfg";
std::string videoPath = absPath_img + "video1.mp4";
std::string lookPath = absPath_img + "blackest.jpg";
bool closeKeyGet = false;
#ifdef KEYBOARD
void mouseHandler(int, int, int, int, void *);
#endif
bool FileExists(const std::string &Filename);
cv::VideoCapture arguments(int argc, char const *argv[]);
int getDisplayInfo();
#ifdef KEYBOARD
void *getKey(void *t_id);
#endif
//********************MAIN**************************************************
int main(int argc, char const *argv[])
{

    cv::VideoCapture cap = arguments(argc, argv); //check if files exist, else exit program
    capture_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    capture_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    //GET SCREEN DISPLAY INFO
    getDisplayInfo();
//OBJECT FOR INFERENCE CLASS AND LOADING THE MODEL
#ifdef DNN
    dnn.load_model(conf, weights, classes);
#endif
#ifdef PROJETION
    //PROJECTION OBJ
    pro.setValues(cap.get(cv::CAP_PROP_FRAME_HEIGHT), cap.get(cv::CAP_PROP_FRAME_WIDTH), screen_width, screen_height);
//create thread for reading keyinputs
#endif
#ifdef KEYBOARD
    pthread_t t;
    int rc = pthread_create(&t, NULL, getKey, (void *)1);
    if (rc)
    {
        std::cout << "Error:unable to create thread," << rc << std::endl;
        exit(-1);
    }
#endif
    //VIDEO CAPTURE
    cv::Mat yoloFrame, projectFrame, frame;
    //for mouse points
    point = cv::Point(0, 0);
#ifdef PROJETION
    cv::Mat mask(pro.getLastSize(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Size lastSize = pro.getLastSize();
#endif

    cv::VideoWriter video("outcpp.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10, cv::Size(capture_width, capture_height));

    while (cap.isOpened() > 0)
    {
        cap >> frame;

        if (frame.empty())
        {
            break;
        }
        frame.copyTo(yoloFrame);
        
//std::cout << "Entry 11" << std::endl;
#ifdef PROJETION
        if (pro.getLastSize() != lastSize)
        {
            cv::resize(mask, mask, pro.getLastSize());
            //std::cout << "Entry 12" << std::endl;
            lastSize = pro.getLastSize();
        }
#endif
//EXCLUDE FROM DETECTION WITH MOUSE SET
//
#ifdef PROJETION
        if (point.x > 0 && point.y > 0)
        {
            //std::cout << "Entry 6" << std::endl;
            cv::circle(mask, point, 2, cv::Scalar(0, 0, 255), -1, 8, 0); //creates a circle point
            //std::cout << "Entry 7" << std::endl;
            pro.setMask(mask);
            //cv::imshow("Mask", mask);
            //std::cout << "Entry 8" << std::endl;
        }
        if (var > 1)
        { //must be 2 points or greater
            //std::cout << "Entry 9" << std::endl;
            cv::line(mask, pts[var - 2], point, cv::Scalar(0, 0, 255), 2, 8, 0);
            //std::cout << "Entry 10" << std::endl;
            pro.setMask(mask);
        }
        if (var != 0 && rightDown)
        {
            std::cout << "SAVED" << std::endl;
            float alfa = 0.5;
            pro.saveMaskPts(pts);
            cv::polylines(mask, pts, 1, cv::Scalar(0, 0, 255), 2, 8, 0);
            pro.setMask(mask);
            rightDown = false;
        }
        if (clearMask)
        {
            pts.clear();
            cv::bitwise_and(mask, cv::Scalar(0, 0, 0), mask);
            clearMask = false;
        }
#endif
        //GET DETECTION BOX ON FRAME
        //std::cout << "Entry 1" << std::endl;
#ifdef DNN
        yoloFrame = dnn.inference(yoloFrame);
        dnn.getDetections(yoloFrame); //get detection vectors
#endif
#ifdef PROJETION
        yoloFrame = pro.Border(yoloFrame); //draw borders
#endif
#ifdef DNN
        yoloFrame = dnn.drawDetection(yoloFrame); //draw detection boxes and labels
#endif
#ifdef PROJETION
        projectFrame = pro.projectionFrame(yoloFrame);
#endif
#ifdef DNN
        yoloFrame = dnn.drawFPS(projectFrame); //Draw FPS
#endif
        // std::cout << "Entry 2" << std::endl;
        //cv::namedWindow("output", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("output", cv::WINDOW_NORMAL);
        cv::setWindowProperty("output", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
#ifdef KEYBOARD
        cv::setMouseCallback("output", mouseHandler, NULL);
#endif
        cv::cvtColor(yoloFrame,yoloFrame,cv::COLOR_RGB2BGR);
        video.write(yoloFrame);
        //std::cout << "Entry 13" << std::endl;
        cv::imshow("output", yoloFrame);
        cv::waitKey(1);
        /*
        if (cv::waitKey(5) >= 0)
            break;
*/
        if (closeKeyGet == true)
        {
            break;
        }
    }
    cap.release();
    video.release();

    pthread_exit(NULL);
    cv::destroyAllWindows();
    return 0;
}

//*********************************************END MAIN**********************

bool FileExists(const std::string &Filename)
{
    bool check = access(Filename.c_str(), 0) == 0;
    return check;
}

cv::VideoCapture arguments(int argc, char const *argv[])
{
    // LETS GET THE FRAME 0 FOR INPUT:2 FOR THERMAL INPUT:0 FOR WEBCAM INPUT:1 FOR VIDEO.MP4 FOLLOWED BY THE PATH
    cv::VideoCapture cap;

    int input = std::stoi(argv[1]);

    //custom
    std::cout << "YOLO LAST CUSTOM" << std::endl;
    absPath_weights = "/home/jav/mss/weights/";
    conf = absPath_weights + "yolov3_obj.cfg";
    weights = absPath_weights + "yolov3_obj.weights";
    classes = absPath_weights + "obj.names";
    //model = "yolov3 cus";

    videoPath = argv[2];

    if (!FileExists(weights) || !FileExists(conf) || !FileExists(videoPath))
    {
        std::cerr << "check which one is missing: " << weights << " " << conf << " " << videoPath << "\n";
        exit(0);
    }
    else
    {
        std::cout << weights << "\n"
                  << conf << "\n"
                  << videoPath << "\n";
    }

    if (argc >= 2 && input == 2)
    {
        printf("Thermal ");
        cap.open(2, cv::CAP_V4L);
    }
    if (argc >= 2 && input == 0)
    {
        printf("webcam ");
        cap.open(0, cv::CAP_V4L);
    }
    if (argc >= 3 && input == 1)
    {

        printf("video\n");

        cap.open(argv[2], cv::CAP_FFMPEG);
    }

    return cap;
}

int getDisplayInfo()
{
    //GET SCREEN RESOLUTION
    Display *display = XOpenDisplay(NULL);

    /* check the number of the default screen for our X server. */
    screen_num = DefaultScreen(display);
    //printf("number of screen %d \n", screen_num);
    /* find the width of the default screen of our X server, in pixels. */
    screen_width = DisplayWidth(display, screen_num);

    /* find the height of the default screen of our X server, in pixels. */
    screen_height = DisplayHeight(display, screen_num);

    /* find the ID of the root window of the screen. */
    root_window = RootWindow(display, screen_num);

    /* find the value of a white pixel on this screen. */
    white_pixel = WhitePixel(display, screen_num);

    /* find the value of a black pixel on this screen. */
    black_pixel = BlackPixel(display, screen_num);
    //printf("width, height display %d %d \n", screen_width, screen_height);
    return 0;
}
#ifdef KEYBOARD
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
                //std::cout << "TOP" << std::endl;
                if (pro.new_y > 0)
                    pro.new_y--;
                break;
            case bottom:
                //std::cout << "BOTTOM" << std::endl;
                if ((pro.new_y + pro.new_frame_h) < pro.black_h)
                    pro.new_y++;
                break;
            case left:
                //std::cout << "LEFT" << std::endl;
                if (pro.new_x > 0)
                    pro.new_x--;
                break;
            case right:
                //std::cout << "RIGHT" << std::endl;
                if ((pro.new_x + pro.new_frame_w) < pro.black_w)
                    pro.new_x++;
                break;
            case scale_up:

                if (pro.new_frame_w < pro.black_w && ((pro.new_frame_w + pro.new_x) + (pro.new_frame_w * ratio)) < pro.black_w && pro.new_frame_h < pro.black_h && ((pro.new_frame_h + pro.new_y) + (pro.new_frame_h * ratio)) < pro.black_h)
                {
                    //std::cout << "scale up" << std::endl;
                    pro.new_frame_w += pro.new_frame_w * ratio;
                    pro.new_frame_h += pro.new_frame_h * ratio;
                    if (pts.size() > 1)
                    {
                        std::vector<cv::Point> tpts;
                        for (auto &&i : pts)
                        {
                            i.x += (i.x * ratio);
                            i.y += (i.y * ratio);
                            tpts.push_back(cv::Point(i.x, i.y));
                        }
                        pts.clear();
                        pts = tpts;
                        //std::cout << "Entry 1" << std::endl;
                        pro.saveMaskPts(pts);
                        // std::cout << "Entry 2" << std::endl;
                        //pro.setMask(mask);
                    }
                }

                break;
            case scale_down:

                if (pro.new_frame_w > (minSizePorce * pro.frame_w))
                {
                    //std::cout << "scale down" << std::endl;
                    pro.new_frame_w -= pro.new_frame_w * ratio;
                    pro.new_frame_h -= pro.new_frame_h * ratio;
                    {
                        if (pts.size() > 1)
                        {
                            std::vector<cv::Point> tpts;
                            for (auto &&i : pts)
                            {
                                i.x -= (i.x * ratio);
                                i.y -= (i.y * ratio);
                                tpts.push_back(cv::Point(i.x, i.y));
                            }
                            pts.clear();
                            pts = tpts;
                            pro.saveMaskPts(pts);
                            //pro.setMask(mask);
                        }
                    }
                }
                break;
            case low_low:

                if (pro.lowerThr > 0)
                {
                    //std::cout << "Low threshold Canny" << std::endl;
                    pro.lowerThr -= 1;
                }

                break;
            case low_high:

                if (pro.lowerThr < 255)
                {
                    //std::cout << "Low threshold Canny" << std::endl;
                    pro.lowerThr += 1;
                }
                break;
            case high_low:

                if (pro.higherThr > 0)
                {
                    //std::cout << "Higher threshold Canny" << std::endl;
                    pro.higherThr -= 1;
                }

                break;
            case high_high:

                if (pro.higherThr < 999)
                {
                    //std::cout << "Higher threshold Canny" << std::endl;
                    pro.higherThr += 1;
                }
                break;
            case esc:
                closeKeyGet = true;
                //exit(0);
                break;

            default:
                std::cout << "KEY NOT MAPPED" << std::endl;
                break;
            }
            pro.saveConfig();
        }
    }
}

void mouseHandler(int event, int x, int y, int, void *)

{
    drag = 0;

    if (event == cv::EVENT_LBUTTONDOWN && !drag) //press left click
    {
        if (flag == 0)
        {

            point.x = x - pro.new_x;
            point.y = y - pro.new_y;

            pts.push_back(point);
            var++;
            drag = 1;
        }
    }
    if (event == cv::EVENT_LBUTTONUP && drag)
    {
        drag = 0;
    }
    if (event == cv::EVENT_RBUTTONDOWN)
    {
        flag = 1;

        if (var != 0)
        {
            rightDown = true;
        }
    }

    if (event == cv::EVENT_RBUTTONUP)
    {
        rightUp = true;
        flag = var;
    }

    if (event == cv::EVENT_MBUTTONDOWN)
    {
        pts.clear();
        var = 0;
        drag = 0;
        flag = 0;

        clearMask = true;
    }
}
#endif