#include "sampleSSD.h"

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the SSD network by parsing the caffe model and builds
//!          the engine that will be used to run SSD (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleSSD::build()
{
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = SampleUniquePtr<nvcaffeparser1::ICaffeParser>(nvcaffeparser1::createCaffeParser());
    if (!parser)
    {
        return false;
    }
    
    auto constructed = constructNetwork(builder, network, config, parser);
   
    if (!constructed)
    {
        return false;
    }

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 3);

    return true;
}

//!
//! \brief Uses a caffe parser to create the SSD Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the SSD network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleSSD::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder> &builder,
                                 SampleUniquePtr<nvinfer1::INetworkDefinition> &network, SampleUniquePtr<nvinfer1::IBuilderConfig> &config,
                                 SampleUniquePtr<nvcaffeparser1::ICaffeParser> &parser)
{

    const nvcaffeparser1::IBlobNameToTensor *blobNameToTensor = parser->parse(locateFile(mParams.prototxtFileName, mParams.dataDirs).c_str(),
                                                                              locateFile(mParams.weightsFileName, mParams.dataDirs).c_str(), *network, DataType::kFLOAT);
   
    for (auto &s : mParams.outputTensorNames)
    {
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }

    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(36_MiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    // Calibrator life time needs to last until after the engine is built.
    std::unique_ptr<IInt8Calibrator> calibrator;

    if (mParams.int8)
    {
        gLogInfo << "Using Entropy Calibrator 2" << std::endl;
        BatchStream calibrationStream(mParams.batchSize, mParams.nbCalBatches, mParams.calibrationBatches, mParams.dataDirs);
        calibrator.reset(new Int8EntropyCalibrator2<BatchStream>(calibrationStream, 0, "SSD", mParams.inputTensorNames[0].c_str()));
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator.get());
    }
    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
cv::Mat SampleSSD::infer(cv::Mat limage)
{
    limage.copyTo(image);
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        std::cout << "ERROR CONTEXT " << std::endl;
    }

    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        std::cout << "ERROR PROCESS INPUT " << std::endl;
    }
    std::cout << "GPU Inner inference " << std::endl;
    auto start = std::chrono::steady_clock::now();
    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->execute(mParams.batchSize, buffers.getDeviceBindings().data());
    if (!status)
    {
        std::cout << "ERROR INFERENCE EXECUTE " << std::endl;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();
    auto end = std::chrono::steady_clock::now();
    auto elapsed = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Inner inference time: " << elapsed << " FPS" << std::endl;
    // Post-process detections and verify results
    if (!verifyOutput(buffers))
    {
        std::cout << "ERROR VERIFY " << std::endl;
    }

    return image;
}

//!
//! \brief Cleans up any state created in the sample class
//!
bool SampleSSD::teardown()
{
    //! Clean up the libprotobuf files as the parsing is complete
    //! \note It is not safe to use any other part of the protocol buffers library after
    //! ShutdownProtobufLibrary() has been called.
    nvcaffeparser1::shutdownProtobufLibrary();
    return true;
}

//!
//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
//!
bool SampleSSD::processInput(const samplesCommon::BufferManager &buffers)
{
    const int inputC = mInputDims.d[0];
    const int inputH = mInputDims.d[1];
    const int inputW = mInputDims.d[2];
    const int batchSize = mParams.batchSize;

    cv::resize(image, image, cv::Size(300, 300));
    vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_PXM_BINARY);
    compression_params.push_back(3);
    //cv::imwrite("image.ppm", image, compression_params);
    cv::imwrite("image.jpeg", image);

    // Load JPEG image
    CImg<unsigned char> im("image.jpeg");

    // Save as PPM
    im.save("image.ppm");

    // Available images

    std::vector<std::string> imageList = {"image.ppm"};

    mPPMs.resize(batchSize);
    assert(mPPMs.size() <= imageList.size());
    for (int i = 0; i < batchSize; ++i)
    {
        readPPMFile(locateFile(imageList[i], mParams.dataDirs), mPPMs[i]);
    }

    // Fill data buffer
    float *hostDataBuffer = static_cast<float *>(buffers.getHostBuffer("data"));
    float pixelMean[3]{104.0f, 117.0f, 123.0f}; // In BGR order
    // Host memory for input buffer
    for (int i = 0, volImg = inputC * inputH * inputW; i < mParams.batchSize; ++i)
    {
        for (int c = 0; c < inputC; ++c)
        {
            // The color image to input should be in BGR order
            for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j)
            {
                hostDataBuffer[i * volImg + c * volChl + j] = float(mPPMs[i].buffer[j * inputC + 2 - c]) - pixelMean[c];
            }
        }
    }
    return true;
}

//!
//! \brief Filters output detections and verify result
//!
//! \return whether the detection output matches expectations
//!
bool SampleSSD::verifyOutput(const samplesCommon::BufferManager &buffers)
{
    const int inputH = mInputDims.d[1];
    const int inputW = mInputDims.d[2];
    const int batchSize = mParams.batchSize;
    const int keepTopK = mParams.keepTopK;
    const float visualThreshold = mParams.visualThreshold;
    const int outputClsSize = mParams.outputClsSize;

    const float *detectionOut = static_cast<const float *>(buffers.getHostBuffer("detection_out"));
    const int *keepCount = static_cast<const int *>(buffers.getHostBuffer("keep_count"));

    const std::vector<std::string> classes{"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"}; // List of class labels

    bool pass = true;

    for (int p = 0; p < batchSize; ++p)
    {
        int numDetections = 0;
        // is there at least one correct detection?
        bool correctDetection = false;
        for (int i = 0; i < keepCount[p]; ++i)
        {
            const float *det = detectionOut + (p * keepTopK + i) * 7;
            if (det[2] < visualThreshold)
            {
                continue;
            }
            assert((int)det[1] < outputClsSize);
            std::string storeName = classes[(int)det[1]] + "-" + std::to_string(det[2]) + ".ppm";

            numDetections++;
            //if (classes[(int)det[1]] == "car")
            //{
            correctDetection = true;
            //}

            gLogInfo << " Image name:" << mPPMs[p].fileName.c_str() << ", Label: " << classes[(int)det[1]].c_str() << ","
                     << " confidence: " << det[2] * 100.f
                     << " xmin: " << det[3] * inputW
                     << " ymin: " << det[4] * inputH
                     << " xmax: " << det[5] * inputW
                     << " ymax: " << det[6] * inputH
                     << std::endl;

            cv::rectangle(image, cv::Point(det[3] * inputW, det[4] * inputH), cv::Point(det[5] * inputW, det[6] * inputH), cv::Scalar(0, 255, 0), 1);

            //samplesCommon::writePPMFileWithBBox(storeName, mPPMs[p], {det[3] * inputW, det[4] * inputH, det[5] * inputW, det[6] * inputH});
        }
        pass &= numDetections >= 1;
        pass &= correctDetection;
    }

    return pass;
}
