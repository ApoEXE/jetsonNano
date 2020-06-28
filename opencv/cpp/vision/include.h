#ifndef INCLUDE_H
#define INCLUDE_H

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

#ifdef _WIN32
#include <io.h>
#define access _access_s
#else
#include <unistd.h>
#endif

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
extern std::string model;

extern std::string absPath_weights;
extern std::string absPath_img;
extern std::string classes;
extern std::string weights;
extern std::string conf;

extern std::string videoPath;
extern std::string lookPath;


#endif