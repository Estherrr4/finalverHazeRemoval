#pragma once
#ifndef IMAGE_PROCESSING_FROM_SCRATCH_DEHAZE_H
#define IMAGE_PROCESSING_FROM_SCRATCH_DEHAZE_H

#include <opencv2/opencv.hpp>
#include <stdexcept>

namespace DarkChannel {
    // Error checking macro
#define __Assert__(x,msg)\
    do\
    {\
        if(!(x)){throw std::runtime_error((msg));}\
    }while(false)

    // Structure to store timing information
    struct TimingInfo {
        double darkChannelTime = 0.0;      
        double atmosphericLightTime = 0.0; 
        double transmissionTime = 0.0;     
        double refinementTime = 0.0;       
        double reconstructionTime = 0.0;   
        double totalTime = 0.0;            
    };

    // Serial implementation
    cv::Mat dehaze(const cv::Mat& img);

    // CUDA implementation
    cv::Mat dehaze_cuda(const cv::Mat& img);

    // Function to get the last timing information
    TimingInfo getLastTimingInfo();

    // Function to check CUDA availability with proper error handling
    bool isCudaAvailable();
}

#endif //IMAGE_PROCESSING_FROM_SCRATCH_DEHAZE_H