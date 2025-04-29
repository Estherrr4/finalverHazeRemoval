#include "dehaze_parallel.h"
#include "fastguidedfilter.h"
#include "dehaze.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <omp.h>

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace DarkChannel {
    extern TimingInfo lastTimingInfo;

    cv::Mat dehazeParallel(const cv::Mat& img, int numThreads) {
        try {
            // Reset timing info
            TimingInfo timingInfo;

            // Start total time measurement
            auto totalStartTime = std::chrono::high_resolution_clock::now();

            // Set number of threads
            omp_set_num_threads(numThreads);

            std::cout << "\n===== OpenMP Performance Timing =====" << std::endl;

            // Check if image is valid
            if (img.empty()) {
                std::cerr << "Error: Input image is empty" << std::endl;
                return img;
            }

            // Check for correct image type
            if (img.type() != CV_8UC3) {
                std::cerr << "Error: Only 3-channel images are supported" << std::endl;
                return img;
            }

            cv::Mat img_double;
            img.convertTo(img_double, CV_64FC3);
            img_double /= 255;

            // Start timing for dark channel calculation
            auto darkChannelStartTime = std::chrono::high_resolution_clock::now();

            // Get dark channel
            int patch_radius = 7;
            cv::Mat darkchannel = cv::Mat::zeros(img_double.rows, img_double.cols, CV_64F);

            // Use OpenMP for parallelism
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < darkchannel.rows; i++) {
                for (int j = 0; j < darkchannel.cols; j++) {
                    int r_start = std::max(0, i - patch_radius);
                    int r_end = std::min(darkchannel.rows, i + patch_radius + 1);
                    int c_start = std::max(0, j - patch_radius);
                    int c_end = std::min(darkchannel.cols, j + patch_radius + 1);
                    double dark = 1.0; // Start with maximum value (after normalization)

                    for (int r = r_start; r < r_end; r++) {
                        for (int c = c_start; c < c_end; c++) {
                            const cv::Vec3d& val = img_double.at<cv::Vec3d>(r, c);
                            dark = std::min(dark, val[0]);
                            dark = std::min(dark, val[1]);
                            dark = std::min(dark, val[2]);
                        }
                    }

                    darkchannel.at<double>(i, j) = dark;
                }
            }

            // End timing for dark channel calculation
            auto darkChannelEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.darkChannelTime = std::chrono::duration<double, std::milli>(darkChannelEndTime - darkChannelStartTime).count();

            // Start timing for atmospheric light estimation
            auto atmosphericStartTime = std::chrono::high_resolution_clock::now();

            // Get atmospheric light
            int pixels = img_double.rows * img_double.cols;
            // Ensure at least 1 pixel is used
            int num = std::max(1, pixels / 1000); // 0.1%
            std::vector<std::pair<double, int>> V;
            V.reserve(pixels);

            // Create flat index for each pixel with its dark channel value
            for (int i = 0; i < darkchannel.rows; i++) {
                for (int j = 0; j < darkchannel.cols; j++) {
                    int index = i * darkchannel.cols + j;
                    V.emplace_back(darkchannel.at<double>(i, j), index);
                }
            }

            // Sort to find brightest pixels in dark channel
            std::sort(V.begin(), V.end(), [](const std::pair<double, int>& p1, const std::pair<double, int>& p2) {
                return p1.first > p2.first;
                });

            // Initialize atmospheric light array
            double atmospheric_light[] = { 0, 0, 0 };

            // Calculate atmospheric light from brightest pixels
            for (int k = 0; k < num; k++) {
                int idx = V[k].second;
                int r = idx / darkchannel.cols;
                int c = idx % darkchannel.cols;
                const cv::Vec3d& val = img_double.at<cv::Vec3d>(r, c);

                atmospheric_light[0] += val[0];
                atmospheric_light[1] += val[1];
                atmospheric_light[2] += val[2];
            }

            atmospheric_light[0] /= num;
            atmospheric_light[1] /= num;
            atmospheric_light[2] /= num;

            // Avoid division by zero
            for (int i = 0; i < 3; i++) {
                if (atmospheric_light[i] < 0.00001) {
                    atmospheric_light[i] = 0.00001;
                }
            }

            // End timing for atmospheric light estimation
            auto atmosphericEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.atmosphericLightTime = std::chrono::duration<double, std::milli>(atmosphericEndTime - atmosphericStartTime).count();

            // Start timing for transmission estimation
            auto transmissionStartTime = std::chrono::high_resolution_clock::now();

            double omega = 0.95;
            cv::Mat channels[3];
            cv::split(img_double, channels);

            // Normalize each channel by atmospheric light
            for (int k = 0; k < 3; k++) {
                channels[k] /= atmospheric_light[k];
            }

            // Merge normalized channels
            cv::Mat temp;
            cv::merge(channels, 3, temp);

            // Get dark channel of normalized image
            cv::Mat temp_dark = cv::Mat::zeros(temp.rows, temp.cols, CV_64F);

            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < temp.rows; i++) {
                for (int j = 0; j < temp.cols; j++) {
                    int r_start = std::max(0, i - patch_radius);
                    int r_end = std::min(temp.rows, i + patch_radius + 1);
                    int c_start = std::max(0, j - patch_radius);
                    int c_end = std::min(temp.cols, j + patch_radius + 1);
                    double dark = 1.0;

                    for (int r = r_start; r < r_end; r++) {
                        for (int c = c_start; c < c_end; c++) {
                            const cv::Vec3d& val = temp.at<cv::Vec3d>(r, c);
                            dark = std::min(dark, val[0]);
                            dark = std::min(dark, val[1]);
                            dark = std::min(dark, val[2]);
                        }
                    }

                    temp_dark.at<double>(i, j) = dark;
                }
            }

            // Get transmission
            cv::Mat transmission = 1.0 - omega * temp_dark;

            // End timing for transmission estimation
            auto transmissionEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.transmissionTime = std::chrono::duration<double, std::milli>(transmissionEndTime - transmissionStartTime).count();

            // Start timing for refinement
            auto refinementStartTime = std::chrono::high_resolution_clock::now();

            // Apply guided filter for refinement
            transmission = fastGuidedFilter(img_double, transmission, 40, 0.1, 5);

            // End timing for refinement
            auto refinementEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.refinementTime = std::chrono::duration<double, std::milli>(refinementEndTime - refinementStartTime).count();

            // Start timing for scene reconstruction
            auto reconstructionStartTime = std::chrono::high_resolution_clock::now();

            // Ensure minimum transmission value to preserve details in dark regions
            double t0 = 0.1;
            cv::Mat trans_bounded;
            cv::max(transmission, t0, trans_bounded);

            // Get a fresh copy of the image channels
            cv::split(img_double, channels);

            // Create result image
            cv::Mat result(img_double.size(), CV_64FC3);
            cv::Vec3d* result_data = (cv::Vec3d*)result.data;

            // Process all pixels in a more straightforward way
            #pragma omp parallel for
            for (int i = 0; i < img_double.rows; i++) {
                for (int j = 0; j < img_double.cols; j++) {
                    const int idx = i * img_double.cols + j;
                    const double t = trans_bounded.at<double>(i, j);
                    const cv::Vec3d& pixel = img_double.at<cv::Vec3d>(i, j);

                    // Apply dehaze formula to each channel: J = (I-A)/t + A
                    cv::Vec3d dehazed;
                    for (int c = 0; c < 3; c++) {
                        dehazed[c] = ((pixel[c] - atmospheric_light[c]) / t) + atmospheric_light[c];
                        // Clamp to valid range
                        dehazed[c] = std::max(0.0, std::min(1.0, dehazed[c]));
                    }

                    result_data[idx] = dehazed;
                }
            }

            // Convert back to 8-bit
            cv::Mat result_8bit;
            result.convertTo(result_8bit, CV_8UC3, 255.0);

            // End timing for scene reconstruction
            auto reconstructionEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.reconstructionTime = std::chrono::duration<double, std::milli>(reconstructionEndTime - reconstructionStartTime).count();

            // End total time measurement
            auto totalEndTime = std::chrono::high_resolution_clock::now();
            timingInfo.totalTime = std::chrono::duration<double, std::milli>(totalEndTime - totalStartTime).count();

            std::cout << std::fixed << std::setprecision(10);
            std::cout << "Dark Channel Calculation: " << timingInfo.darkChannelTime << " ms" << std::endl;
            std::cout << "Atmospheric Light Estimation: " << timingInfo.atmosphericLightTime << " ms" << std::endl;
            std::cout << "Transmission Estimation: " << timingInfo.transmissionTime << " ms" << std::endl;
            std::cout << "Transmission Refinement: " << timingInfo.refinementTime << " ms" << std::endl;
            std::cout << "Scene Reconstruction: " << timingInfo.reconstructionTime << " ms" << std::endl;
            std::cout << "Total Execution Time: " << timingInfo.totalTime << " ms" << std::endl;
            std::cout << "========================================" << std::endl;

            lastTimingInfo = timingInfo;

            return result_8bit;
        }
        catch (const cv::Exception& e) {
            std::cerr << "OpenCV Exception in OpenMP implementation: " << e.what() << std::endl;
            return img;
        }
        catch (const std::exception& e) {
            std::cerr << "Standard Exception in OpenMP implementation: " << e.what() << std::endl;
            return img;
        }
        catch (...) {
            std::cerr << "Unknown exception in dehazeParallel function" << std::endl;
            return img;
        }
    }

    bool isCudaAvailable();
}