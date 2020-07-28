//g++ -std=c++11 -O3 -march=native -Dcimg_jpeg=1 -Dcimg_display=0 sampleSSD.cpp common/*.cpp -o test -lcudnn -lcublas -lcudart -lnvinfer -lnvparsers -lnvinfer_plugin -lnvonnxparser -I./common -I/usr/local/cuda/include -L/usr/local/cuda/lib64 `pkg-config opencv --cflags --libs`

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "BatchStream.h"
#include "EntropyCalibrator.h"

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include "CImg.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <fstream>
#include <chrono>
#include <string>

using namespace cimg_library;
const std::string gSampleName = "TensorRT.sample_ssd";

//!
//! \brief The SampleSSDParams structure groups the additional parameters required by
//!         the SSD sample.
//!
struct SampleSSDParams : public samplesCommon::CaffeSampleParams
{
    int outputClsSize;              //!< The number of output classes
    int keepTopK;                   //!< The maximum number of detection post-NMS
    int nbCalBatches;               //!< The number of batches for calibration
    float visualThreshold;          //!< The minimum score threshold to consider a detection
    std::string calibrationBatches; //!< The path to calibration batches
};

//! \brief  The SampleSSD class implements the SSD sample
//!
//! \details It creates the network using a caffe model
//!
class SampleSSD
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleSSD(const SampleSSDParams &params)
        : mParams(params), mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    cv::Mat infer(cv::Mat);

    //!
    //! \brief Cleans up any state created in the sample class
    //!
    bool teardown();

private:
    cv::Mat image;
    SampleSSDParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.

    std::vector<samplesCommon::PPM<3, 300, 300>> mPPMs; //!< PPMs of test images

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses a Caffe model for SSD and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder> &builder,
                          SampleUniquePtr<nvinfer1::INetworkDefinition> &network, SampleUniquePtr<nvinfer1::IBuilderConfig> &config,
                          SampleUniquePtr<nvcaffeparser1::ICaffeParser> &parser);

    //!
    //! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager &buffers);

    //!
    //! \brief Filters output detections and verify results
    //!
    bool verifyOutput(const samplesCommon::BufferManager &buffers);
};
