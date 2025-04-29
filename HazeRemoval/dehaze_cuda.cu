#include "dehaze.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <stdio.h>

// Define CUDA_ENABLED for isCudaAvailable() function
#define CUDA_ENABLED

// Improved error checking macro
#define checkCudaErrors(val) { \
    cudaError_t err = (val); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl; \
        std::cerr << "Error code: " << err << std::endl; \
        throw std::runtime_error("CUDA error"); \
    } \
}

// Kernel error checking macro (for after kernel launches)
#define checkKernelErrors() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl; \
        std::cerr << "Error code: " << err << std::endl; \
        throw std::runtime_error("Kernel launch error"); \
    } \
    err = cudaDeviceSynchronize(); \
    if (err != cudaSuccess) { \
        std::cerr << "Kernel execution error: " << cudaGetErrorString(err) << " at " \
                  << __FILE__ << ":" << __LINE__ << std::endl; \
        std::cerr << "Error code: " << err << std::endl; \
        throw std::runtime_error("Kernel execution error"); \
    } \
}

//Check CUDA Part
// Define a simple kernel
__global__ void testKernel() {
    printf("Inside CUDA kernel!\n");
}

// Define a function to launch the kernel
extern "C" void launchCudaPart() {
    try {
        printf("Launching CUDA kernel from launchCudaPart()...\n");

        // Check if device is available
        int deviceCount;
        checkCudaErrors(cudaGetDeviceCount(&deviceCount));

        if (deviceCount == 0) {
            printf("No CUDA devices found!\n");
            return;
        }

        // Get device properties
        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
        printf("Using CUDA device: %s\n", deviceProp.name);

        // Launch test kernel
        testKernel << <1, 1 >> > ();
        checkKernelErrors();

        printf("Test kernel completed successfully!\n");
    }
    catch (const std::exception& e) {
        printf("CUDA test failed: %s\n", e.what());
    }
}
//Check CUDA Part End

// Kernel to compute dark channel
__global__ void darkChannelKernel(const unsigned char* imgData, float* darkChannel, int width, int height, int channels, int patch_radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float minVal = 1.0f;
    int idx = y * width + x;

    // Search through patch around current pixel
    for (int dy = -patch_radius; dy <= patch_radius; dy++) {
        for (int dx = -patch_radius; dx <= patch_radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int nidx = (ny * width + nx) * channels;

                // Find minimum across RGB channels
                float b = imgData[nidx] / 255.0f;
                float g = imgData[nidx + 1] / 255.0f;
                float r = imgData[nidx + 2] / 255.0f;

                float pixelMin = fminf(r, fminf(g, b));
                minVal = fminf(minVal, pixelMin);
            }
        }
    }

    darkChannel[idx] = minVal;
}

// Kernel to compute normalized dark channel for transmission
__global__ void transmissionKernel(const unsigned char* imgData, const float* atmospheric, float* transmission, int width, int height, int channels, int patch_radius, float omega) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float minVal = 1.0f;
    int idx = y * width + x;

    // Search through patch around current pixel
    for (int dy = -patch_radius; dy <= patch_radius; dy++) {
        for (int dx = -patch_radius; dx <= patch_radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int nidx = (ny * width + nx) * channels;

                // Normalize by atmospheric light and find minimum
                float b = imgData[nidx] / 255.0f / atmospheric[0];
                float g = imgData[nidx + 1] / 255.0f / atmospheric[1];
                float r = imgData[nidx + 2] / 255.0f / atmospheric[2];

                float pixelMin = fminf(r, fminf(g, b));
                minVal = fminf(minVal, pixelMin);
            }
        }
    }

    // Calculate transmission
    transmission[idx] = 1.0f - omega * minVal;
}

// Improved kernel for scene reconstruction with better color handling
__global__ void sceneRecoveryKernel(const unsigned char* imgData, const float* transmission, const float* atmospheric, unsigned char* outputData, int width, int height, int channels, float t0) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;
    int tidx = y * width + x;

    // Get transmission value with lower bound t0
    float t = fmaxf(transmission[tidx], t0);

    // Process each channel separately to ensure color is preserved
    for (int c = 0; c < 3; c++) {
        float normalized = imgData[idx + c] / 255.0f;
        float recovered = (normalized - atmospheric[c]) / t + atmospheric[c];

        // Clamp to [0, 1] range
        recovered = fminf(fmaxf(recovered, 0.0f), 1.0f);

        // Convert back to 8-bit
        outputData[idx + c] = static_cast<unsigned char>(recovered * 255.0f);
    }
}

namespace DarkChannel {
    extern TimingInfo lastTimingInfo;

    bool isCudaAvailable() {
        try {
            int deviceCount = 0;
            cudaError_t error = cudaGetDeviceCount(&deviceCount);
            if (error != cudaSuccess) {
                std::cerr << "CUDA device check failed: " << cudaGetErrorString(error) << std::endl;
                return false;
            }

            if (deviceCount == 0) {
                std::cerr << "No CUDA devices found on this system." << std::endl;
                return false;
            }

            // Additional check: verify device capabilities
            cudaDeviceProp deviceProp;
            checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));

            // Check for compute capability (minimum 3.0 required)
            if (deviceProp.major < 3) {
                std::cerr << "CUDA device has compute capability "
                    << deviceProp.major << "." << deviceProp.minor
                    << ", but 3.0 or higher is required." << std::endl;
                return false;
            }

            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Exception checking CUDA availability: " << e.what() << std::endl;
            return false;
        }
    }

    cv::Mat dehaze_cuda(const cv::Mat& img) {
        try {
            // Reset timing info
            TimingInfo timingInfo;

            // Start total time measurement
            auto totalStartTime = std::chrono::high_resolution_clock::now();

            if (!isCudaAvailable()) {
                std::cerr << "No suitable CUDA device found. Falling back to CPU implementation." << std::endl;
                return dehaze(img);
            }

            cudaDeviceProp deviceProp;
            checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
            std::cout << "Using CUDA device: " << deviceProp.name << " with compute capability "
                << deviceProp.major << "." << deviceProp.minor << std::endl;

            // Check if image is valid
            if (img.empty() || img.type() != CV_8UC3) {
                std::cerr << "Error: Input image is empty or not 3-channel 8-bit" << std::endl;
                return img;
            }

            // Image dimensions
            int width = img.cols;
            int height = img.rows;
            int channels = img.channels();
            int imgSize = width * height;
            size_t imgBytes = imgSize * channels * sizeof(unsigned char);

            std::cout << "Processing image size: " << width << "x" << height << " ("
                << (imgBytes / (1024 * 1024)) << " MB)" << std::endl;

            // Allocate host and device memory
            unsigned char* d_img = nullptr;
            float* d_darkChannel = nullptr;
            float* d_transmission = nullptr;
            unsigned char* d_result = nullptr;
            float* d_atmospheric = nullptr;

            try {
                // Start dark channel calculation timing
                auto darkChannelStartTime = std::chrono::high_resolution_clock::now();

                // Allocate device memory
                checkCudaErrors(cudaMalloc(&d_img, imgBytes));
                checkCudaErrors(cudaMalloc(&d_darkChannel, imgSize * sizeof(float)));

                // Copy input image to device
                checkCudaErrors(cudaMemcpy(d_img, img.data, imgBytes, cudaMemcpyHostToDevice));

                // Define block and grid dimensions - adjust for large images
                // Use smaller block size for better occupancy with large images
                dim3 blockSize(16, 16);
                dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

                // Dark channel calculation
                int patchRadius = 7;
                darkChannelKernel << <gridSize, blockSize >> > (d_img, d_darkChannel, width, height, channels, patchRadius);
                checkKernelErrors();

                // Copy dark channel back to host for atmospheric light estimation
                float* h_darkChannel = new float[imgSize];
                checkCudaErrors(cudaMemcpy(h_darkChannel, d_darkChannel, imgSize * sizeof(float), cudaMemcpyDeviceToHost));

                // End dark channel timing
                auto darkChannelEndTime = std::chrono::high_resolution_clock::now();
                timingInfo.darkChannelTime = std::chrono::duration<double, std::milli>(darkChannelEndTime - darkChannelStartTime).count();

                // Start atmospheric light estimation timing
                auto atmosphericStartTime = std::chrono::high_resolution_clock::now();

                // Find atmospheric light (top 0.1% brightest pixels in dark channel)
                std::vector<std::pair<float, int>> brightnessIndices;
                brightnessIndices.reserve(imgSize);

                for (int i = 0; i < imgSize; i++) {
                    brightnessIndices.push_back(std::make_pair(h_darkChannel[i], i));
                }

                std::sort(brightnessIndices.begin(), brightnessIndices.end(),
                    [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                        return a.first > b.first;
                    });

                // Take top 0.1% of pixels
                int numBrightestPixels = std::max(1, imgSize / 1000);
                float atmospheric[3] = { 0, 0, 0 };

                for (int i = 0; i < numBrightestPixels; i++) {
                    int idx = brightnessIndices[i].second;
                    int y = idx / width;
                    int x = idx % width;

                    cv::Vec3b pixelValue = img.at<cv::Vec3b>(y, x);
                    atmospheric[0] += pixelValue[0] / 255.0f;  // B
                    atmospheric[1] += pixelValue[1] / 255.0f;  // G
                    atmospheric[2] += pixelValue[2] / 255.0f;  // R
                }

                atmospheric[0] /= numBrightestPixels;
                atmospheric[1] /= numBrightestPixels;
                atmospheric[2] /= numBrightestPixels;

                // Avoid division by zero
                for (int i = 0; i < 3; i++) {
                    if (atmospheric[i] < 0.00001f) {
                        atmospheric[i] = 0.00001f;
                    }
                }

                // End atmospheric light timing
                auto atmosphericEndTime = std::chrono::high_resolution_clock::now();
                timingInfo.atmosphericLightTime = std::chrono::duration<double, std::milli>(atmosphericEndTime - atmosphericStartTime).count();

                // Start transmission estimation timing
                auto transmissionStartTime = std::chrono::high_resolution_clock::now();

                // Allocate memory for transmission map
                checkCudaErrors(cudaMalloc(&d_transmission, imgSize * sizeof(float)));

                // Copy atmospheric light to device
                checkCudaErrors(cudaMalloc(&d_atmospheric, 3 * sizeof(float)));
                checkCudaErrors(cudaMemcpy(d_atmospheric, atmospheric, 3 * sizeof(float), cudaMemcpyHostToDevice));

                // Estimate transmission using dark channel prior
                float omega = 0.95f;  // Preserves some haze for depth perception
                transmissionKernel << <gridSize, blockSize >> > (d_img, d_atmospheric, d_transmission, width, height, channels, patchRadius, omega);
                checkKernelErrors();

                // End transmission timing
                auto transmissionEndTime = std::chrono::high_resolution_clock::now();
                timingInfo.transmissionTime = std::chrono::duration<double, std::milli>(transmissionEndTime - transmissionStartTime).count();

                // Start refinement timing
                auto refinementStartTime = std::chrono::high_resolution_clock::now();

                // For simplicity, we'll skip guided filter refinement in the CUDA version
                // In a real implementation, you would implement the guided filter as CUDA kernels

                // End refinement timing
                auto refinementEndTime = std::chrono::high_resolution_clock::now();
                timingInfo.refinementTime = std::chrono::duration<double, std::milli>(refinementEndTime - refinementStartTime).count();

                // Start scene reconstruction timing
                auto reconstructionStartTime = std::chrono::high_resolution_clock::now();

                // Allocate memory for result
                checkCudaErrors(cudaMalloc(&d_result, imgBytes));

                // Recover scene
                float t0 = 0.1f;  // Minimum transmission value
                sceneRecoveryKernel << <gridSize, blockSize >> > (d_img, d_transmission, d_atmospheric, d_result, width, height, channels, t0);
                checkKernelErrors();

                // Copy result back to host
                cv::Mat result(height, width, CV_8UC3);
                checkCudaErrors(cudaMemcpy(result.data, d_result, imgBytes, cudaMemcpyDeviceToHost));

                // Verify the image is valid
                if (result.empty()) {
                    std::cerr << "Error: CUDA result is empty after memory copy" << std::endl;
                    throw std::runtime_error("Invalid CUDA result");
                }

                // Verify the image has the correct dimensions
                if (result.rows != height || result.cols != width || result.channels() != 3) {
                    std::cerr << "Error: CUDA result has invalid dimensions or channel count" << std::endl;
                    std::cerr << "Expected: " << width << "x" << height << "x3, Got: "
                        << result.cols << "x" << result.rows << "x" << result.channels() << std::endl;
                    throw std::runtime_error("Invalid CUDA result dimensions");
                }

                std::cout << "CUDA processing complete. Result dimensions: "
                    << result.cols << "x" << result.rows << "x" << result.channels() << std::endl;

                // End scene reconstruction timing
                auto reconstructionEndTime = std::chrono::high_resolution_clock::now();
                timingInfo.reconstructionTime = std::chrono::duration<double, std::milli>(reconstructionEndTime - reconstructionStartTime).count();

                // End total time measurement
                auto totalEndTime = std::chrono::high_resolution_clock::now();
                timingInfo.totalTime = std::chrono::duration<double, std::milli>(totalEndTime - totalStartTime).count();

                std::cout << "\n===== CUDA Performance Timing (milliseconds) =====" << std::endl;
                std::cout << std::fixed << std::setprecision(10);
                std::cout << "Dark Channel Calculation: " << timingInfo.darkChannelTime << " ms" << std::endl;
                std::cout << "Atmospheric Light Estimation: " << timingInfo.atmosphericLightTime << " ms" << std::endl;
                std::cout << "Transmission Estimation: " << timingInfo.transmissionTime << " ms" << std::endl;
                std::cout << "Transmission Refinement: " << timingInfo.refinementTime << " ms" << std::endl;
                std::cout << "Scene Reconstruction: " << timingInfo.reconstructionTime << " ms" << std::endl;
                std::cout << "Total Execution Time: " << timingInfo.totalTime << " ms" << std::endl;
                std::cout << "========================================" << std::endl;

                // Save timing info in the global variable for access
                lastTimingInfo = timingInfo;

                // Cleanup
                delete[] h_darkChannel;
                cudaFree(d_img);
                cudaFree(d_darkChannel);
                cudaFree(d_transmission);
                cudaFree(d_atmospheric);
                cudaFree(d_result);

                return result;
            }
            catch (const std::exception& e) {
                std::cerr << "Exception during CUDA processing: " << e.what() << std::endl;

                // Cleanup - free allocated memory to avoid memory leaks
                if (d_img) cudaFree(d_img);
                if (d_darkChannel) cudaFree(d_darkChannel);
                if (d_transmission) cudaFree(d_transmission);
                if (d_atmospheric) cudaFree(d_atmospheric);
                if (d_result) cudaFree(d_result);

                std::cerr << "Falling back to CPU implementation." << std::endl;
                return dehaze(img);
            }
        }
        catch (const cv::Exception& e) {
            std::cerr << "OpenCV Exception in CUDA implementation: " << e.what() << std::endl;
            return img;
        }
        catch (const std::exception& e) {
            std::cerr << "Standard Exception in CUDA implementation: " << e.what() << std::endl;
            return img;
        }
        catch (...) {
            std::cerr << "Unknown exception in dehaze_cuda function" << std::endl;
            return img;
        }
    }
}