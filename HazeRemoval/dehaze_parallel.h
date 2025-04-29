#pragma once
#ifndef DEHAZE_PARALLEL_H
#define DEHAZE_PARALLEL_H

#include <opencv2/opencv.hpp>

namespace DarkChannel {
    // Non-threaded OpenMP - using vectorization
    cv::Mat dehazeParallel(const cv::Mat& img, int numThreads = 1);

    // Check CUDA availability
    bool isCudaAvailable();
}

#endif // DEHAZE_PARALLEL_H