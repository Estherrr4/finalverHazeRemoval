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

// Optimized constant memory for performance
__constant__ float c_eps = 0.1f;
__constant__ int c_window_radius = 8; // For box filter

// Improved dark channel kernel with shared memory optimization
__global__ void darkChannelKernel(const unsigned char* imgData, float* darkChannel, int width, int height, int channels, int patch_radius) {
    extern __shared__ unsigned char sharedImg[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Local thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Shared memory block size with halo cells
    int block_width = blockDim.x + 2 * patch_radius;
    int block_height = blockDim.y + 2 * patch_radius;

    // Load image data into shared memory, including halo
    for (int dy = -patch_radius; dy <= patch_radius; dy += blockDim.y) {
        for (int dx = -patch_radius; dx <= patch_radius; dx += blockDim.x) {
            int sx = tx + dx + patch_radius;
            int sy = ty + dy + patch_radius;

            // Check if within shared memory bounds
            if (sx >= 0 && sx < block_width && sy >= 0 && sy < block_height) {
                int imgX = x + dx;
                int imgY = y + dy;

                // Check if within image bounds
                if (imgX >= 0 && imgX < width && imgY >= 0 && imgY < height) {
                    int shared_idx = (sy * block_width + sx) * channels;
                    int img_idx = (imgY * width + imgX) * channels;

                    // Copy RGB values
                    for (int c = 0; c < channels; c++) {
                        sharedImg[shared_idx + c] = imgData[img_idx + c];
                    }
                }
                else {
                    // Out of bounds, use a default value
                    int shared_idx = (sy * block_width + sx) * channels;
                    for (int c = 0; c < channels; c++) {
                        sharedImg[shared_idx + c] = 255; // Not dark
                    }
                }
            }
        }
    }

    __syncthreads();

    // Only process within image bounds
    if (x >= width || y >= height) return;

    // Calculate dark channel for this pixel
    float minVal = 1.0f;

    for (int dy = -patch_radius; dy <= patch_radius; dy++) {
        for (int dx = -patch_radius; dx <= patch_radius; dx++) {
            int sx = tx + dx + patch_radius;
            int sy = ty + dy + patch_radius;

            // Check if within shared memory bounds
            if (sx >= 0 && sx < block_width && sy >= 0 && sy < block_height) {
                int shared_idx = (sy * block_width + sx) * channels;

                // Find minimum across RGB channels
                float b = sharedImg[shared_idx] / 255.0f;
                float g = sharedImg[shared_idx + 1] / 255.0f;
                float r = sharedImg[shared_idx + 2] / 255.0f;

                float pixelMin = fminf(r, fminf(g, b));
                minVal = fminf(minVal, pixelMin);
            }
        }
    }

    darkChannel[y * width + x] = minVal;
}

// Optimized transmission estimation kernel
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

    // Calculate transmission with a more conservative omega
    transmission[idx] = 1.0f - (omega * 0.9f) * minVal;
}

// Sharp box filter implementation for guided filter
__global__ void boxFilterKernel(const float* input, float* output, int width, int height, int radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const int idx = y * width + x;
    float sum = 0.0f;
    int count = 0;

    // Calculate the box filter by directly summing values
    for (int j = max(0, y - radius); j <= min(height - 1, y + radius); j++) {
        for (int i = max(0, x - radius); i <= min(width - 1, x + radius); i++) {
            sum += input[j * width + i];
            count++;
        }
    }

    // Normalize
    output[idx] = sum / count;
}

// Subtraction kernel
__global__ void subtractKernel(const float* a, const float* b, float* result, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const int idx = y * width + x;
    result[idx] = a[idx] - b[idx];
}

// Addition kernel
__global__ void addKernel(const float* a, const float* b, float* result, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const int idx = y * width + x;
    result[idx] = a[idx] + b[idx];
}

// Multiplication kernel
__global__ void multiplyKernel(const float* a, const float* b, float* result, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const int idx = y * width + x;
    result[idx] = a[idx] * b[idx];
}

// Division kernel with regularization
__global__ void divideKernel(const float* a, const float* b, float* result, int width, int height, float eps) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const int idx = y * width + x;
    result[idx] = a[idx] / (b[idx] + eps);
}

// Improved bilinear upsampling kernel with edge preservation
__global__ void upsampleKernel(const float* input, float* output, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstWidth || y >= dstHeight) return;

    // Map destination coordinates to source coordinates
    float srcX = x * ((float)srcWidth / dstWidth);
    float srcY = y * ((float)srcHeight / dstHeight);

    // Get the four surrounding pixels
    int x0 = floor(srcX);
    int y0 = floor(srcY);
    int x1 = min(x0 + 1, srcWidth - 1);
    int y1 = min(y0 + 1, srcHeight - 1);

    // Calculate interpolation weights
    float wx = srcX - x0;
    float wy = srcY - y0;

    // Perform bilinear interpolation
    float top = (1.0f - wx) * input[y0 * srcWidth + x0] + wx * input[y0 * srcWidth + x1];
    float bottom = (1.0f - wx) * input[y1 * srcWidth + x0] + wx * input[y1 * srcWidth + x1];
    float value = (1.0f - wy) * top + wy * bottom;

    // Store the result
    output[y * dstWidth + x] = value;
}

// Convert RGB to grayscale
__global__ void rgbToGrayKernel(const unsigned char* rgb, float* gray, int width, int height, int channels) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const int idx = y * width + x;
    const int rgbIdx = idx * channels;

    // Convert RGB to grayscale using standard weights
    gray[idx] = (0.299f * rgb[rgbIdx + 2] + 0.587f * rgb[rgbIdx + 1] + 0.114f * rgb[rgbIdx]) / 255.0f;
}

// Downsampling kernel
__global__ void downsampleKernel(const float* input, float* output, int srcWidth, int srcHeight, int dstWidth, int dstHeight) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstWidth || y >= dstHeight) return;

    // Calculate region to average in source image
    float scaleX = (float)srcWidth / dstWidth;
    float scaleY = (float)srcHeight / dstHeight;

    int startX = floor(x * scaleX);
    int endX = floor((x + 1) * scaleX);
    int startY = floor(y * scaleY);
    int endY = floor((y + 1) * scaleY);

    // Ensure valid bounds
    startX = max(0, startX);
    endX = min(srcWidth - 1, endX);
    startY = max(0, startY);
    endY = min(srcHeight - 1, endY);

    // Average all pixels in the region
    float sum = 0.0f;
    int count = 0;

    for (int j = startY; j <= endY; j++) {
        for (int i = startX; i <= endX; i++) {
            sum += input[j * srcWidth + i];
            count++;
        }
    }

    output[y * dstWidth + x] = sum / max(count, 1);
}

// Apply guided filter
__global__ void applyFilterKernel(const float* a, const float* b, float* output, float* gray, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const int idx = y * width + x;

    // Apply linear transformation: a*I + b
    output[idx] = a[idx] * gray[idx] + b[idx];

    // Clamp to [0, 1] range
    output[idx] = fmaxf(0.0f, fminf(1.0f, output[idx]));
}

// Sharpen kernel for enhancing details
__global__ void sharpenKernel(const float* input, float* output, int width, int height, float strength) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height || x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        // Skip border pixels, just copy
        if (x < width && y < height) {
            output[y * width + x] = input[y * width + x];
        }
        return;
    }

    const int idx = y * width + x;

    // Simple Laplacian filter
    float center = input[idx];
    float top = input[(y - 1) * width + x];
    float bottom = input[(y + 1) * width + x];
    float left = input[y * width + (x - 1)];
    float right = input[y * width + (x + 1)];

    // Apply sharpening: center + strength * (center - average of neighbors)
    float laplacian = 4.0f * center - top - bottom - left - right;
    output[idx] = center + strength * laplacian;

    // Clamp result to [0, 1]
    output[idx] = fmaxf(0.0f, fminf(1.0f, output[idx]));
}

// Enhanced scene recovery with better quality and sharpness
__global__ void sceneRecoveryKernel(const unsigned char* imgData, const float* transmission,
    const float* atmospheric, unsigned char* outputData,
    int width, int height, int channels, float t0) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;
    int tidx = y * width + x;

    // Higher minimum transmission to preserve detail in dark areas
    float t = fmaxf(transmission[tidx], t0);

    // Use adaptive tone mapping to prevent over-darkening
    float intensity = 0.0f;
    for (int c = 0; c < channels; c++) {
        intensity += imgData[idx + c] / 255.0f;
    }
    intensity /= channels;

    // Adaptive t0 based on intensity and spatial location
    float adaptive_t0 = t0;
    if (intensity < 0.3f) {
        // For darker regions, use higher minimum transmission
        adaptive_t0 = t0 * 1.5f;
        t = fmaxf(t, adaptive_t0);
    }

    // For sky areas (typically upper portion of image), preserve more haze
    if (y < height / 3 && intensity > 0.7f) {
        t = fmaxf(t, 0.4f); // Preserve some atmospheric effect for sky
    }

    // Process each channel separately with enhanced color preservation
    for (int c = 0; c < channels; c++) {
        float normalized = imgData[idx + c] / 255.0f;

        // Modified recovery formula with better exposure control
        float recovered = (normalized - atmospheric[c]) / t + atmospheric[c];

        // Enhance brightness for dark scenes (gamma correction)
        recovered = powf(recovered, 0.85f);

        // Apply additional contrast but keep reasonable limits
        recovered = (recovered - 0.5f) * 1.2f + 0.5f;

        // Clamp to [0, 1] range
        recovered = fminf(fmaxf(recovered, 0.0f), 1.0f);

        // Convert back to 8-bit
        outputData[idx + c] = static_cast<unsigned char>(recovered * 255.0f);
    }
}

// Guide filter implementation for CUDA with improved sharpness
void guidedFilterCuda(const unsigned char* d_I, float* d_p, float* d_output,
    int width, int height, int channels, int r, float eps, int s) {
    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 fullGridSize((width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);

    // Calculate small dimensions - smaller subsample factor for better quality
    s = min(s, 3); // Limit subsample factor to maintain quality
    int smallWidth = width / s;
    int smallHeight = height / s;
    dim3 smallGridSize((smallWidth + blockSize.x - 1) / blockSize.x,
        (smallHeight + blockSize.y - 1) / blockSize.y);

    // Allocate memory for intermediate results
    float* d_I_gray = nullptr;
    float* d_I_small = nullptr;
    float* d_p_small = nullptr;
    float* d_mean_I = nullptr;
    float* d_mean_p = nullptr;
    float* d_mean_Ip = nullptr;
    float* d_mean_II = nullptr;
    float* d_var_I = nullptr;
    float* d_cov_Ip = nullptr;
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_mean_a = nullptr;
    float* d_mean_b = nullptr;
    float* d_a_big = nullptr;
    float* d_b_big = nullptr;
    float* d_temp = nullptr;
    float* d_sharpened = nullptr;

    // Allocate device memory
    checkCudaErrors(cudaMalloc(&d_I_gray, width * height * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_I_small, smallWidth * smallHeight * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_p_small, smallWidth * smallHeight * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_mean_I, smallWidth * smallHeight * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_mean_p, smallWidth * smallHeight * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_mean_Ip, smallWidth * smallHeight * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_mean_II, smallWidth * smallHeight * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_var_I, smallWidth * smallHeight * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_cov_Ip, smallWidth * smallHeight * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_a, smallWidth * smallHeight * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_b, smallWidth * smallHeight * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_mean_a, smallWidth * smallHeight * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_mean_b, smallWidth * smallHeight * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_a_big, width * height * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_b_big, width * height * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_temp, smallWidth * smallHeight * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_sharpened, width * height * sizeof(float)));

    // Convert input image to grayscale
    rgbToGrayKernel << <fullGridSize, blockSize >> > (d_I, d_I_gray, width, height, channels);
    checkKernelErrors();

    // Downsample grayscale image
    downsampleKernel << <smallGridSize, blockSize >> > (d_I_gray, d_I_small, width, height, smallWidth, smallHeight);
    checkKernelErrors();

    // Downsample transmission map
    downsampleKernel << <smallGridSize, blockSize >> > (d_p, d_p_small, width, height, smallWidth, smallHeight);
    checkKernelErrors();

    // Mean of I
    boxFilterKernel << <smallGridSize, blockSize >> > (d_I_small, d_mean_I, smallWidth, smallHeight, r / s);
    checkKernelErrors();

    // Mean of p
    boxFilterKernel << <smallGridSize, blockSize >> > (d_p_small, d_mean_p, smallWidth, smallHeight, r / s);
    checkKernelErrors();

    // Mean of I * p
    multiplyKernel << <smallGridSize, blockSize >> > (d_I_small, d_p_small, d_temp, smallWidth, smallHeight);
    boxFilterKernel << <smallGridSize, blockSize >> > (d_temp, d_mean_Ip, smallWidth, smallHeight, r / s);
    checkKernelErrors();

    // Mean of I * I
    multiplyKernel << <smallGridSize, blockSize >> > (d_I_small, d_I_small, d_temp, smallWidth, smallHeight);
    boxFilterKernel << <smallGridSize, blockSize >> > (d_temp, d_mean_II, smallWidth, smallHeight, r / s);
    checkKernelErrors();

    // Variance of I
    multiplyKernel << <smallGridSize, blockSize >> > (d_mean_I, d_mean_I, d_temp, smallWidth, smallHeight);
    subtractKernel << <smallGridSize, blockSize >> > (d_mean_II, d_temp, d_var_I, smallWidth, smallHeight);
    checkKernelErrors();

    // Covariance of I and p
    multiplyKernel << <smallGridSize, blockSize >> > (d_mean_I, d_mean_p, d_temp, smallWidth, smallHeight);
    subtractKernel << <smallGridSize, blockSize >> > (d_mean_Ip, d_temp, d_cov_Ip, smallWidth, smallHeight);
    checkKernelErrors();

    // Calculate a and b
    // Smaller epsilon for better edge preservation
    float adjusted_eps = eps * 0.5f;
    divideKernel << <smallGridSize, blockSize >> > (d_cov_Ip, d_var_I, d_a, smallWidth, smallHeight, adjusted_eps);
    multiplyKernel << <smallGridSize, blockSize >> > (d_a, d_mean_I, d_temp, smallWidth, smallHeight);
    subtractKernel << <smallGridSize, blockSize >> > (d_mean_p, d_temp, d_b, smallWidth, smallHeight);
    checkKernelErrors();

    // Mean of a and b with smaller radius for better edge preservation
    int refinement_radius = r / s - 1;
    refinement_radius = max(1, refinement_radius);
    boxFilterKernel << <smallGridSize, blockSize >> > (d_a, d_mean_a, smallWidth, smallHeight, refinement_radius);
    boxFilterKernel << <smallGridSize, blockSize >> > (d_b, d_mean_b, smallWidth, smallHeight, refinement_radius);
    checkKernelErrors();

    // Upsample a and b to original resolution
    upsampleKernel << <fullGridSize, blockSize >> > (d_mean_a, d_a_big, smallWidth, smallHeight, width, height);
    upsampleKernel << <fullGridSize, blockSize >> > (d_mean_b, d_b_big, smallWidth, smallHeight, width, height);
    checkKernelErrors();

    // Apply filter: q = a*I + b
    applyFilterKernel << <fullGridSize, blockSize >> > (d_a_big, d_b_big, d_output, d_I_gray, width, height);
    checkKernelErrors();

    // Apply sharpening to enhance details
    sharpenKernel << <fullGridSize, blockSize >> > (d_output, d_sharpened, width, height, 0.3f);
    checkKernelErrors();

    // Copy sharpened result back to output
    checkCudaErrors(cudaMemcpy(d_output, d_sharpened, width * height * sizeof(float), cudaMemcpyDeviceToDevice));

    // Free memory
    cudaFree(d_I_gray);
    cudaFree(d_I_small);
    cudaFree(d_p_small);
    cudaFree(d_mean_I);
    cudaFree(d_mean_p);
    cudaFree(d_mean_Ip);
    cudaFree(d_mean_II);
    cudaFree(d_var_I);
    cudaFree(d_cov_Ip);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_mean_a);
    cudaFree(d_mean_b);
    cudaFree(d_a_big);
    cudaFree(d_b_big);
    cudaFree(d_temp);
    cudaFree(d_sharpened);
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
            float* d_refined_transmission = nullptr;
            unsigned char* d_result = nullptr;
            float* d_atmospheric = nullptr;
            float* d_sharpened = nullptr;

            try {
                // Start dark channel calculation timing
                auto darkChannelStartTime = std::chrono::high_resolution_clock::now();

                // Allocate device memory
                checkCudaErrors(cudaMalloc(&d_img, imgBytes));
                checkCudaErrors(cudaMalloc(&d_darkChannel, imgSize * sizeof(float)));
                checkCudaErrors(cudaMalloc(&d_sharpened, imgSize * sizeof(float)));

                // Copy input image to device
                checkCudaErrors(cudaMemcpy(d_img, img.data, imgBytes, cudaMemcpyHostToDevice));

                // Define block and grid dimensions - adjust for large images
                dim3 blockSize(16, 16);
                dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

                // Calculate shared memory size for the dark channel kernel
                int patchRadius = 7;
                int sharedWidth = blockSize.x + 2 * patchRadius;
                int sharedHeight = blockSize.y + 2 * patchRadius;
                size_t sharedMemSize = sharedWidth * sharedHeight * channels * sizeof(unsigned char);

                // Dark channel calculation with shared memory
                darkChannelKernel << <gridSize, blockSize, sharedMemSize >> > (d_img, d_darkChannel, width, height, channels, patchRadius);
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

                // Use top 0.1% brightest pixels but add protections
                int numBrightestPixels = std::max(1, imgSize / 1000);
                float atmospheric[3] = { 0, 0, 0 };
                float maxAtmospheric = 0.0f;
                bool validAtmospheric = false;

                // First pass - find the maximum value
                for (int i = 0; i < numBrightestPixels; i++) {
                    int idx = brightnessIndices[i].second;
                    int y = idx / width;
                    int x = idx % width;

                    cv::Vec3b pixelValue = img.at<cv::Vec3b>(y, x);
                    float avgIntensity = (pixelValue[0] + pixelValue[1] + pixelValue[2]) / 3.0f;

                    if (avgIntensity > maxAtmospheric) {
                        maxAtmospheric = avgIntensity;
                    }
                }

                // Now use pixels that are not too far from the max
                for (int i = 0; i < numBrightestPixels; i++) {
                    int idx = brightnessIndices[i].second;
                    int y = idx / width;
                    int x = idx % width;

                    cv::Vec3b pixelValue = img.at<cv::Vec3b>(y, x);
                    float avgIntensity = (pixelValue[0] + pixelValue[1] + pixelValue[2]) / 3.0f;

                    // Only use pixels that are at least 70% as bright as the brightest
                    if (avgIntensity > maxAtmospheric * 0.7f) {
                        atmospheric[0] += pixelValue[0] / 255.0f;  // B
                        atmospheric[1] += pixelValue[1] / 255.0f;  // G
                        atmospheric[2] += pixelValue[2] / 255.0f;  // R
                        validAtmospheric = true;
                    }
                }

                // Normalize if we found valid pixels
                if (validAtmospheric) {
                    atmospheric[0] /= numBrightestPixels;
                    atmospheric[1] /= numBrightestPixels;
                    atmospheric[2] /= numBrightestPixels;
                }
                else {
                    // Use a fallback if no good atmospheric light found
                    atmospheric[0] = 0.8f;
                    atmospheric[1] = 0.8f;
                    atmospheric[2] = 0.8f;
                }

                // Ensure atmospheric light is not too dark - critical for hazy scenes
                float minAtmosphericThreshold = 0.7f;
                for (int i = 0; i < 3; i++) {
                    if (atmospheric[i] < minAtmosphericThreshold) {
                        atmospheric[i] = minAtmosphericThreshold;
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

                // Estimate transmission using dark channel prior with a conservative omega
                float omega = 0.75f;  // Less aggressive to preserve natural look
                transmissionKernel << <gridSize, blockSize >> > (d_img, d_atmospheric, d_transmission, width, height, channels, patchRadius, omega);
                checkKernelErrors();

                // End transmission timing
                auto transmissionEndTime = std::chrono::high_resolution_clock::now();
                timingInfo.transmissionTime = std::chrono::duration<double, std::milli>(transmissionEndTime - transmissionStartTime).count();

                // Start refinement timing
                auto refinementStartTime = std::chrono::high_resolution_clock::now();

                // Allocate memory for refined transmission
                checkCudaErrors(cudaMalloc(&d_refined_transmission, imgSize * sizeof(float)));

                // Apply guided filter with improved edge preservation
                guidedFilterCuda(d_img, d_transmission, d_refined_transmission, width, height, channels, 30, 0.05f, 2);  // Smaller radius and epsilon for better detail

                // End refinement timing
                auto refinementEndTime = std::chrono::high_resolution_clock::now();
                timingInfo.refinementTime = std::chrono::duration<double, std::milli>(refinementEndTime - refinementStartTime).count();

                // Start scene reconstruction timing
                auto reconstructionStartTime = std::chrono::high_resolution_clock::now();

                // Allocate memory for result
                checkCudaErrors(cudaMalloc(&d_result, imgBytes));

                // Recover scene with enhanced quality and adaptive parameters
                float t0 = 0.1f;  // Minimum transmission value
                sceneRecoveryKernel << <gridSize, blockSize >> > (d_img, d_refined_transmission, d_atmospheric, d_result, width, height, channels, t0);
                checkKernelErrors();

                // Copy result back to host
                cv::Mat result(height, width, CV_8UC3);
                checkCudaErrors(cudaMemcpy(result.data, d_result, imgBytes, cudaMemcpyDeviceToHost));

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

                // Apply post-processing for better visual quality and sharpness
                cv::Mat enhancedResult;
                result.convertTo(enhancedResult, CV_32F, 1.0 / 255.0);

                // Sharpen the image using unsharp mask
                cv::Mat blurred;
                cv::GaussianBlur(enhancedResult, blurred, cv::Size(0, 0), 1.5);
                cv::Mat unsharpMask = enhancedResult - blurred;
                enhancedResult = enhancedResult + 0.7 * unsharpMask;  // Adjust strength as needed

                // Apply adaptive contrast enhancement
                cv::Scalar meanValue = cv::mean(enhancedResult);
                float meanIntensity = (meanValue[0] + meanValue[1] + meanValue[2]) / 3.0f;

                // Adjust brightness if image is too dark
                if (meanIntensity < 0.4f) {
                    float brightnessAdjust = 0.4f / meanIntensity;
                    brightnessAdjust = std::min(brightnessAdjust, 1.5f);  // Limit adjustment
                    enhancedResult = enhancedResult * brightnessAdjust;
                }

                // Apply additional contrast enhancement
                cv::Mat contrastEnhanced;
                enhancedResult.convertTo(contrastEnhanced, -1, 1.1, 0.05);  // Increase contrast slightly

                // Apply detail-preserving bilateral filter to reduce noise while keeping edges sharp
                cv::Mat bilateralFiltered;
                cv::bilateralFilter(contrastEnhanced, bilateralFiltered, 5, 10, 10);

                // Apply final sharpening pass
                cv::Mat sharpened;
                cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
                    0, -0.5, 0,
                    -0.5, 3, -0.5,
                    0, -0.5, 0);
                cv::filter2D(bilateralFiltered, sharpened, -1, kernel);

                // Convert back to 8-bit for output
                sharpened.convertTo(result, CV_8UC3, 255.0);

                // Cleanup
                delete[] h_darkChannel;
                cudaFree(d_img);
                cudaFree(d_darkChannel);
                cudaFree(d_transmission);
                cudaFree(d_refined_transmission);
                cudaFree(d_atmospheric);
                cudaFree(d_result);
                cudaFree(d_sharpened);

                return result;
            }
            catch (const std::exception& e) {
                std::cerr << "Exception during CUDA processing: " << e.what() << std::endl;

                // Cleanup - free allocated memory to avoid memory leaks
                if (d_img) cudaFree(d_img);
                if (d_darkChannel) cudaFree(d_darkChannel);
                if (d_transmission) cudaFree(d_transmission);
                if (d_refined_transmission) cudaFree(d_refined_transmission);
                if (d_atmospheric) cudaFree(d_atmospheric);
                if (d_result) cudaFree(d_result);
                if (d_sharpened) cudaFree(d_sharpened);

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