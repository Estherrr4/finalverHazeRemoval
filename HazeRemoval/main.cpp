#include "dehaze.h"
#include "dehaze_parallel.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <omp.h>

#ifdef _WIN32
#include <Windows.h>
#include <commdlg.h>
#include <conio.h> // For _getch() on Windows
#endif

inline int get_max(int a, int b) {
    return (a > b) ? a : b;
}

// Simple function to check if a file exists
bool fileExists(const std::string& filename) {
    std::ifstream f(filename.c_str());
    return f.good();
}

// Function to wait for a key press without blocking the return to menu
void waitForKeyPress() {
    std::cout << "\nPress any key to return to menu..." << std::endl;

#ifdef _WIN32
    _getch(); // Use _getch() on Windows which doesn't require Enter key
#else
    // For non-Windows platforms
    getchar();
#endif
}

// Function to show file open dialog (ANSI version)
std::string getImageFile() {
#ifdef _WIN32
    OPENFILENAMEA ofn;
    char fileName[MAX_PATH] = "";
    ZeroMemory(&ofn, sizeof(ofn));

    ofn.lStructSize = sizeof(OPENFILENAMEA);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = "Image Files (*.jpg;*.jpeg;*.png;*.bmp)\0*.jpg;*.jpeg;*.png;*.bmp\0All Files (*.*)\0*.*\0";
    ofn.lpstrFile = fileName;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
    ofn.lpstrDefExt = "jpg";

    if (GetOpenFileNameA(&ofn)) {
        return std::string(fileName);
    }
    return "";
#else
    // For non-Windows platforms, use simple input
    std::string filePath;
    std::cout << "Enter path to image file: ";
    std::getline(std::cin, filePath);
    return filePath;
#endif
}

// Function to resize large images while maintaining aspect ratio
cv::Mat resizeImageIfTooLarge(const cv::Mat& img, int maxDimension = 1200) {
    // Check if the image is too large
    if (img.cols > maxDimension || img.rows > maxDimension) {
        cv::Mat resized;
        double scale = (double)maxDimension / (img.cols > img.rows ? img.cols : img.rows);
        int newWidth = static_cast<int>(img.cols * scale);
        int newHeight = static_cast<int>(img.rows * scale);

        cv::resize(img, resized, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_AREA);
        std::cout << "Image resized from " << img.cols << "x" << img.rows
            << " to " << resized.cols << "x" << resized.rows << std::endl;
        return resized;
    }
    return img;
}

// Function to process a single image
bool processImage(const std::string& img_path, const std::string& project_path) {
    try {
        // Use forward slashes consistently
        std::string normalized_path = img_path;
        std::replace(normalized_path.begin(), normalized_path.end(), '\\', '/');

        std::cout << "Processing image: " << normalized_path << std::endl;

        // Load image
        cv::Mat original_img = cv::imread(normalized_path);
        if (original_img.empty()) {
            std::cerr << "Error: Failed to load image: " << normalized_path << std::endl;
            waitForKeyPress();
            return false;
        }

        // Extract just the filename for the output file
        size_t pos = normalized_path.find_last_of('/');
        std::string img_name = (pos != std::string::npos) ? normalized_path.substr(pos + 1) : normalized_path;

        std::cout << "Image loaded successfully. Original size: " << original_img.cols << "x" << original_img.rows << std::endl;

        // Resize large images for faster processing
        cv::Mat img = resizeImageIfTooLarge(original_img, 1200);

        // Save original and resized version of input image for reference
        std::string original_path = project_path + "original_" + img_name;
        cv::imwrite(original_path, original_img);

        if (img.data != original_img.data) {
            std::string resized_input_path = project_path + "resized_input_" + img_name;
            cv::imwrite(resized_input_path, img);
            std::cout << "Original image saved as: " << original_path << std::endl;
            std::cout << "Resized input saved as: " << resized_input_path << std::endl;
        }

        // Process with serial version
        std::cout << "\nRunning serial version..." << std::endl;
        cv::Mat serial_result = DarkChannel::dehaze(img);
        DarkChannel::TimingInfo serial_timing = DarkChannel::getLastTimingInfo();

        // Process with OpenMP version
        std::cout << "\nRunning OpenMP version..." << std::endl;
        int num_threads = get_max(1, omp_get_max_threads());
        std::cout << "Using " << num_threads << " OpenMP threads" << std::endl;
        cv::Mat openmp_result = DarkChannel::dehazeParallel(img, num_threads);
        DarkChannel::TimingInfo openmp_timing = DarkChannel::getLastTimingInfo();

        // Check CUDA availability
        bool cuda_available = DarkChannel::isCudaAvailable();
        cv::Mat cuda_result;
        DarkChannel::TimingInfo cuda_timing;

        if (cuda_available) {
            // Run CUDA implementation if available
            std::cout << "\nRunning CUDA version..." << std::endl;
            try {
                cuda_result = DarkChannel::dehaze_cuda(img);
                cuda_timing = DarkChannel::getLastTimingInfo();

                // Verify the CUDA result is valid
                if (cuda_result.empty()) {
                    std::cerr << "Warning: CUDA result is empty" << std::endl;
                    cuda_available = false;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "CUDA processing failed: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU implementation." << std::endl;
                cuda_available = false;
            }
        }
        else {
            std::cout << "\nCUDA implementation not available." << std::endl;
        }

        std::cout << "\n===== DEHAZING PERFORMANCE COMPARISON (milliseconds) =====" << std::endl;
        std::cout << std::left << std::setw(25) << "\nStage";
        std::cout << std::right << std::setw(17) << "Serial";
        std::cout << std::setw(17) << "OpenMP";
        if (cuda_available) std::cout << std::setw(17) << "CUDA";
        std::cout << std::setw(17) << "OMP Speedup";
        if (cuda_available) std::cout << std::setw(17) << "CUDA Speedup";
        std::cout << std::endl;

        int dividerWidth = 25 + 17 + 17 + 17 + (cuda_available ? 17 + 19 : 0);
        std::cout << std::string(dividerWidth, '-') << std::endl;

        std::cout << std::fixed << std::setprecision(10);

        std::cout << std::left << std::setw(25) << "Dark Channel Calculation";
        std::cout << std::right << std::setw(17) << serial_timing.darkChannelTime;
        std::cout << std::setw(17) << openmp_timing.darkChannelTime;

        if (cuda_available) {
            std::cout << std::setw(17) << cuda_timing.darkChannelTime;
        }

        double omp_speedup = (serial_timing.darkChannelTime > 0) ?
        (serial_timing.darkChannelTime / openmp_timing.darkChannelTime) : 0.0;
        std::cout << std::setw(17) << omp_speedup;

        if (cuda_available) {
            double cuda_speedup = (serial_timing.darkChannelTime > 0) ?
                (serial_timing.darkChannelTime / cuda_timing.darkChannelTime) : 0.0;
            std::cout << std::setw(19) << cuda_speedup;
        }
        std::cout << std::endl;

        std::cout << std::left << std::setw(25) << "Atmospheric Light";
        std::cout << std::right << std::setw(17) << serial_timing.atmosphericLightTime;
        std::cout << std::setw(17) << openmp_timing.atmosphericLightTime;

        if (cuda_available) {
            std::cout << std::setw(17) << cuda_timing.atmosphericLightTime;
        }

        omp_speedup = (serial_timing.atmosphericLightTime > 0) ?
        (serial_timing.atmosphericLightTime / openmp_timing.atmosphericLightTime) : 0.0;
        std::cout << std::setw(17) << omp_speedup;

        if (cuda_available) {
            double cuda_speedup = (serial_timing.atmosphericLightTime > 0) ?
                (serial_timing.atmosphericLightTime / cuda_timing.atmosphericLightTime) : 0.0;
            std::cout << std::setw(19) << cuda_speedup;
        }
        std::cout << std::endl;

        std::cout << std::left << std::setw(25) << "Transmission Estimation";
        std::cout << std::right << std::setw(17) << serial_timing.transmissionTime;
        std::cout << std::setw(17) << openmp_timing.transmissionTime;

        if (cuda_available) {
            std::cout << std::setw(17) << cuda_timing.transmissionTime;
        }

        omp_speedup = (serial_timing.transmissionTime > 0) ?
        (serial_timing.transmissionTime / openmp_timing.transmissionTime) : 0.0;
        std::cout << std::setw(17) << omp_speedup;

        if (cuda_available) {
            double cuda_speedup = (serial_timing.transmissionTime > 0) ?
                (serial_timing.transmissionTime / cuda_timing.transmissionTime) : 0.0;
            std::cout << std::setw(19) << cuda_speedup;
        }
        std::cout << std::endl;

        std::cout << std::left << std::setw(25) << "Transmission Refinement";
        std::cout << std::right << std::setw(17) << serial_timing.refinementTime;
        std::cout << std::setw(17) << openmp_timing.refinementTime;

        if (cuda_available) {
            std::cout << std::setw(17) << cuda_timing.refinementTime;
        }

        omp_speedup = (serial_timing.refinementTime > 0) ?
        (serial_timing.refinementTime / openmp_timing.refinementTime) : 0.0;
        std::cout << std::setw(17) << omp_speedup;

        if (cuda_available) {
            double cuda_speedup = (serial_timing.refinementTime > 0) ?
                (serial_timing.refinementTime / cuda_timing.refinementTime) : 0.0;
            std::cout << std::setw(19) << cuda_speedup;
        }
        std::cout << std::endl;

        std::cout << std::left << std::setw(25) << "Scene Reconstruction";
        std::cout << std::right << std::setw(17) << serial_timing.reconstructionTime;
        std::cout << std::setw(17) << openmp_timing.reconstructionTime;

        if (cuda_available) {
            std::cout << std::setw(17) << cuda_timing.reconstructionTime;
        }

        omp_speedup = (serial_timing.reconstructionTime > 0) ?
        (serial_timing.reconstructionTime / openmp_timing.reconstructionTime) : 0.0;
        std::cout << std::setw(17) << omp_speedup;

        if (cuda_available) {
            double cuda_speedup = (serial_timing.reconstructionTime > 0) ?
                (serial_timing.reconstructionTime / cuda_timing.reconstructionTime) : 0.0;
            std::cout << std::setw(19) << cuda_speedup;
        }
        std::cout << std::endl;

        std::cout << std::left << std::setw(25) << "Total Execution Time";
        std::cout << std::right << std::setw(17) << serial_timing.totalTime;
        std::cout << std::setw(17) << openmp_timing.totalTime;

        if (cuda_available) {
            std::cout << std::setw(17) << cuda_timing.totalTime;
        }

        omp_speedup = (serial_timing.totalTime > 0) ?
        (serial_timing.totalTime / openmp_timing.totalTime) : 0.0;
        std::cout << std::setw(17) << omp_speedup;

        if (cuda_available) {
            double cuda_speedup = (serial_timing.totalTime > 0) ?
                (serial_timing.totalTime / cuda_timing.totalTime) : 0.0;
            std::cout << std::setw(19) << cuda_speedup;
        }
        std::cout << std::endl;

        std::cout << std::string(dividerWidth, '=') << std::endl;


        // Save results
        std::string serial_output_path = project_path + "serial_dehazed_" + img_name;
        std::string openmp_output_path = project_path + "openmp_dehazed_" + img_name;
        std::string cuda_output_path;

        cv::imwrite(serial_output_path, serial_result);
        cv::imwrite(openmp_output_path, openmp_result);

        if (cuda_available) {
            cuda_output_path = project_path + "cuda_dehazed_" + img_name;
            cv::imwrite(cuda_output_path, cuda_result);
        }

        std::cout << "Results saved as: " << std::endl;
        std::cout << "  Serial: " << serial_output_path << std::endl;
        std::cout << "  OpenMP: " << openmp_output_path << std::endl;
        if (cuda_available) {
            std::cout << "  CUDA: " << cuda_output_path << std::endl;
        }

        // Create copies for display to avoid modifying original data
        cv::Mat display_img = img.clone();
        cv::Mat display_serial = serial_result.clone();
        cv::Mat display_openmp = openmp_result.clone();
        cv::Mat display_cuda;
        if (cuda_available && !cuda_result.empty()) {
            display_cuda = cuda_result.clone();
        }

        // Check if result images are valid
        std::cout << "\nValidating result images..." << std::endl;
        std::cout << "Original: " << display_img.cols << "x" << display_img.rows
            << " channels: " << display_img.channels() << std::endl;
        std::cout << "Serial: " << display_serial.cols << "x" << display_serial.rows
            << " channels: " << display_serial.channels() << std::endl;
        std::cout << "OpenMP: " << display_openmp.cols << "x" << display_openmp.rows
            << " channels: " << display_openmp.channels() << std::endl;

        if (cuda_available && !display_cuda.empty()) {
            std::cout << "CUDA: " << display_cuda.cols << "x" << display_cuda.rows
                << " channels: " << display_cuda.channels() << std::endl;
        }

        // Calculate display size based on orientation
        int maxDisplayDimension = 800;
        bool isPortrait = (img.rows > img.cols);

        int displayWidth, displayHeight;
        double scale;

        if (isPortrait) {
            // For portrait images, use height as the limiting factor
            scale = (double)maxDisplayDimension / img.rows;
            displayHeight = maxDisplayDimension;
            displayWidth = static_cast<int>(img.cols * scale);
        }
        else {
            // For landscape images, use width as the limiting factor
            scale = (double)maxDisplayDimension / img.cols;
            displayWidth = maxDisplayDimension;
            displayHeight = static_cast<int>(img.rows * scale);
        }

        // Resize images for display
        if (display_img.cols != displayWidth || display_img.rows != displayHeight) {
            cv::resize(display_img, display_img, cv::Size(displayWidth, displayHeight), 0, 0, cv::INTER_AREA);
            cv::resize(display_serial, display_serial, cv::Size(displayWidth, displayHeight), 0, 0, cv::INTER_AREA);
            cv::resize(display_openmp, display_openmp, cv::Size(displayWidth, displayHeight), 0, 0, cv::INTER_AREA);

            if (cuda_available && !display_cuda.empty()) {
                cv::resize(display_cuda, display_cuda, cv::Size(displayWidth, displayHeight), 0, 0, cv::INTER_AREA);
            }

            std::cout << "Images resized for display to " << displayWidth << "x" << displayHeight << std::endl;
        }

        // Check intensity and normalize if too dark
        auto checkAndNormalize = [](cv::Mat& image, const std::string& name) {
            if (!image.empty()) {
                cv::Scalar meanIntensity = cv::mean(image);
                double avgIntensity = (meanIntensity[0] + meanIntensity[1] + meanIntensity[2]) / 3.0;
                std::cout << name << " result average intensity: " << avgIntensity << std::endl;

                if (avgIntensity < 30.0) {
                    std::cout << name << " result appears dark, applying normalization..." << std::endl;
                    cv::Mat temp;
                    cv::normalize(image, temp, 0, 255, cv::NORM_MINMAX);
                    image = temp;
                }
            }
            };

        checkAndNormalize(display_openmp, "OpenMP");
        if (cuda_available && !display_cuda.empty()) {
            checkAndNormalize(display_cuda, "CUDA");
        }

        cv::namedWindow("Original", cv::WINDOW_NORMAL);
        cv::imshow("Original", display_img);
        cv::resizeWindow("Original", displayWidth, displayHeight);
        cv::moveWindow("Original", 50, 50);
        cv::waitKey(100);  // Small delay to ensure window is created and visible


        cv::namedWindow("Serial Result", cv::WINDOW_NORMAL);
        cv::imshow("Serial Result", display_serial);
        cv::resizeWindow("Serial Result", displayWidth, displayHeight);
        cv::moveWindow("Serial Result", 50 + displayWidth + 20, 50);
        cv::waitKey(100);  // Small delay


        cv::namedWindow("OpenMP Result", cv::WINDOW_NORMAL);
        cv::imshow("OpenMP Result", display_openmp);
        cv::resizeWindow("OpenMP Result", displayWidth, displayHeight);

        // Position based on orientation
        if (isPortrait) {
            cv::moveWindow("OpenMP Result", 50, 50 + displayHeight + 40);
        }
        else {
            cv::moveWindow("OpenMP Result", 50, 50 + displayHeight + 20);
        }
        cv::waitKey(100);  


        if (cuda_available && !display_cuda.empty()) {
            cv::namedWindow("CUDA Result", cv::WINDOW_NORMAL);
            cv::imshow("CUDA Result", display_cuda);
            cv::resizeWindow("CUDA Result", displayWidth, displayHeight);

            if (isPortrait) {
                cv::moveWindow("CUDA Result", 50 + displayWidth + 20, 50 + displayHeight + 40);
            }
            else {
                cv::moveWindow("CUDA Result", 50 + displayWidth + 20, 50 + displayHeight + 20);
            }
            cv::waitKey(100);  
        }

        cv::waitKey(1);

        std::cout << "\nViewing results. Press any key in any image window to continue..." << std::endl;
        cv::waitKey(0);

        if (cuda_available && !display_cuda.empty()) {
            cv::destroyWindow("CUDA Result");
            cv::waitKey(50);  // Small delay
        }
        cv::destroyWindow("OpenMP Result");
        cv::waitKey(50);
        cv::destroyWindow("Serial Result");
        cv::waitKey(50);
        cv::destroyWindow("Original");
        cv::waitKey(50);

        cv::destroyAllWindows();
        cv::waitKey(50);

        waitForKeyPress();

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in processImage: " << e.what() << std::endl;
        waitForKeyPress();
        return false;
    }
}


void exportPerformanceDataToCSV(
    const std::string& filename,
    const std::vector<std::string>& imagePaths,
    const std::vector<std::pair<DarkChannel::TimingInfo, DarkChannel::TimingInfo>>& timingPairs,
    bool cudaAvailable = false,
    const std::vector<DarkChannel::TimingInfo>* cudaTimings = nullptr) {

    std::ofstream csvFile(filename);
    if (!csvFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    csvFile << "Image,ImageSize,Orientation,Stage,Serial_Time,OpenMP_Time,OpenMP_Speedup";
    if (cudaAvailable) {
        csvFile << ",CUDA_Time,CUDA_Speedup";
    }
    csvFile << std::endl;


    for (size_t i = 0; i < imagePaths.size(); i++) {
        cv::Mat img = cv::imread(imagePaths[i]);
        std::string orientation = (img.rows > img.cols) ? "Portrait" : "Landscape";
        std::string imageSize = std::to_string(img.cols) + "x" + std::to_string(img.rows);
        std::string imageName = imagePaths[i].substr(imagePaths[i].find_last_of("/\\") + 1);

        const DarkChannel::TimingInfo& serial = timingPairs[i].first;
        const DarkChannel::TimingInfo& openmp = timingPairs[i].second;


        auto writeRow = [&](const std::string& stage, double serialTime, double openmpTime) {
            double openmpSpeedup = (serialTime > 0) ? (serialTime / openmpTime) : 0.0;

            csvFile << imageName << "," << imageSize << "," << orientation << ","
                << stage << "," << serialTime << "," << openmpTime << ","
                << openmpSpeedup;

            if (cudaAvailable && cudaTimings) {
                double cudaTime = 0.0;

                if (stage == "Dark Channel") {
                    cudaTime = (*cudaTimings)[i].darkChannelTime;
                }
                else if (stage == "Atmospheric Light") {
                    cudaTime = (*cudaTimings)[i].atmosphericLightTime;
                }
                else if (stage == "Transmission") {
                    cudaTime = (*cudaTimings)[i].transmissionTime;
                }
                else if (stage == "Refinement") {
                    cudaTime = (*cudaTimings)[i].refinementTime;
                }
                else if (stage == "Reconstruction") {
                    cudaTime = (*cudaTimings)[i].reconstructionTime;
                }
                else if (stage == "Total") {
                    cudaTime = (*cudaTimings)[i].totalTime;
                }

                double cudaSpeedup = (serialTime > 0) ? (serialTime / cudaTime) : 0.0;
                csvFile << "," << cudaTime << "," << cudaSpeedup;
            }

            csvFile << std::endl;
            };

        writeRow("Dark Channel", serial.darkChannelTime, openmp.darkChannelTime);
        writeRow("Atmospheric Light", serial.atmosphericLightTime, openmp.atmosphericLightTime);
        writeRow("Transmission", serial.transmissionTime, openmp.transmissionTime);
        writeRow("Refinement", serial.refinementTime, openmp.refinementTime);
        writeRow("Reconstruction", serial.reconstructionTime, openmp.reconstructionTime);
        writeRow("Total", serial.totalTime, openmp.totalTime);
    }

    csvFile.close();
    std::cout << "Performance data exported to " << filename << std::endl;
}

// Declare the CUDA test function
extern "C" void launchCudaPart();

int main(int argc, char** argv) {
    try {
        // Test CUDA functionality
        std::cout << "Testing CUDA functionality..." << std::endl;
        try {
            launchCudaPart();
            std::cout << "CUDA test completed." << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "CUDA test failed: " << e.what() << std::endl;
            std::cerr << "Some features may be unavailable." << std::endl;
        }

        std::string project_path = "./";

        bool interactive_mode = true;
        std::string img_path;

        if (argc > 1) {
            interactive_mode = false;
            std::string arg_path = argv[1];

            if (arg_path.find(':') == std::string::npos) {
                img_path = project_path + arg_path;
            }
            else {
                img_path = arg_path;
            }

            processImage(img_path, project_path);

            interactive_mode = true;
        }

        if (interactive_mode) {
            int choice = -1;
            while (choice != 0) {
                std::cout << "\n=== Dehaze Interactive Mode ===" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  1: Enter image path manually" << std::endl;
                std::cout << "  2: Use file dialog to select image" << std::endl;
                std::cout << "  3: Process multiple images" << std::endl;
                std::cout << "  4: Check CUDA availability" << std::endl;
                std::cout << "  5: Run performance benchmark (for Colab plotting)" << std::endl;
                std::cout << "  0: Exit" << std::endl;

                std::cout << "\nEnter your choice (0-5): ";
                std::cin >> choice;
                std::cin.ignore(); 

                switch (choice) {
                case 0:
                    std::cout << "Exiting program." << std::endl;
                    break;

                case 1: {
                    // Manual path entry
                    std::cout << "Enter image path (or 'exit' to return to menu): ";
                    std::getline(std::cin, img_path);
                    if (img_path == "exit" || img_path == "quit") {
                        break;
                    }
                    processImage(img_path, project_path);
                    break;
                }

                case 2: {
                    // File dialog
                    img_path = getImageFile();
                    if (!img_path.empty()) {
                        processImage(img_path, project_path);
                    }
                    else {
                        std::cout << "No file selected." << std::endl;
                        waitForKeyPress();
                    }
                    break;
                }

                case 3: {
                    // Process multiple images
                    std::cout << "Enter 'exit' at any time to return to menu." << std::endl;
                    while (true) {
                        std::cout << "\nSelect option for next image:" << std::endl;
                        std::cout << "  1: Enter path manually" << std::endl;
                        std::cout << "  2: Use file dialog" << std::endl;
                        std::cout << "  0: Return to main menu" << std::endl;

                        int subChoice;
                        std::cout << "Choice: ";
                        std::cin >> subChoice;
                        std::cin.ignore(); 

                        if (subChoice == 0) {
                            break;
                        }
                        else if (subChoice == 1) {
                            std::cout << "Enter image path: ";
                            std::getline(std::cin, img_path);
                            if (img_path == "exit" || img_path == "quit") {
                                break;
                            }
                            processImage(img_path, project_path);
                        }
                        else if (subChoice == 2) {
                            img_path = getImageFile();
                            if (img_path.empty()) {
                                std::cout << "No file selected." << std::endl;
                                waitForKeyPress();
                                continue;
                            }
                            processImage(img_path, project_path);
                        }
                        else {
                            std::cout << "Invalid choice." << std::endl;
                            waitForKeyPress();
                            continue;
                        }
                    }
                    break;
                }

                case 4: {
                    // Check CUDA availability details
                    std::cout << "Checking CUDA availability..." << std::endl;
                    bool cuda_available = DarkChannel::isCudaAvailable();

                    if (cuda_available) {
                        std::cout << "CUDA is available and working correctly." << std::endl;

                        // Get more detailed CUDA info
                        int deviceCount = 0;
                        cudaGetDeviceCount(&deviceCount);

                        std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

                        for (int i = 0; i < deviceCount; i++) {
                            cudaDeviceProp deviceProp;
                            cudaGetDeviceProperties(&deviceProp, i);

                            std::cout << "\nDevice " << i << ": " << deviceProp.name << std::endl;
                            std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
                            std::cout << "  Total global memory: " << (deviceProp.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
                            std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
                            std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
                            std::cout << "  Max threads dimensions: ("
                                << deviceProp.maxThreadsDim[0] << ", "
                                << deviceProp.maxThreadsDim[1] << ", "
                                << deviceProp.maxThreadsDim[2] << ")" << std::endl;
                            std::cout << "  Max grid dimensions: ("
                                << deviceProp.maxGridSize[0] << ", "
                                << deviceProp.maxGridSize[1] << ", "
                                << deviceProp.maxGridSize[2] << ")" << std::endl;
                        }
                    }
                    else {
                        std::cout << "CUDA is not available or not functioning correctly." << std::endl;
                    }

                    waitForKeyPress();
                    break;
                }

                case 5: {
                    // Performance benchmarking mode
                    std::cout << "\n=== Performance Benchmarking Mode ===" << std::endl;
                    std::cout << "This will process multiple images and export performance data for plotting." << std::endl;

                    int numImages;
                    std::cout << "Enter number of images to benchmark: ";
                    std::cin >> numImages;
                    std::cin.ignore(); 

                    if (numImages <= 0) {
                        std::cout << "Invalid number. Returning to main menu." << std::endl;
                        waitForKeyPress();
                        break;
                    }

                    std::vector<std::string> imagePaths;
                    std::vector<std::pair<DarkChannel::TimingInfo, DarkChannel::TimingInfo>> timingPairs;
                    std::vector<DarkChannel::TimingInfo> cudaTimings;

                    for (int i = 0; i < numImages; i++) {
                        std::cout << "\nImage " << (i + 1) << " of " << numImages << ":" << std::endl;
                        std::cout << "  1: Enter path manually" << std::endl;
                        std::cout << "  2: Use file dialog" << std::endl;

                        int imgChoice;
                        std::cout << "Choice: ";
                        std::cin >> imgChoice;
                        std::cin.ignore();

                        std::string imgPath;
                        if (imgChoice == 1) {
                            std::cout << "Enter image path: ";
                            std::getline(std::cin, imgPath);
                        }
                        else if (imgChoice == 2) {
                            imgPath = getImageFile();
                            if (imgPath.empty()) {
                                std::cout << "No file selected. Skipping this image." << std::endl;
                                i--; 
                                continue;
                            }
                        }
                        else {
                            std::cout << "Invalid choice. Skipping this image." << std::endl;
                            i--; 
                            continue;
                        }

                        // Process the image silently
                        std::cout << "Processing image: " << imgPath << std::endl;

                        cv::Mat img = cv::imread(imgPath);
                        if (img.empty()) {
                            std::cerr << "Error: Failed to load image: " << imgPath << std::endl;
                            i--; 
                            continue;
                        }

                        cv::Mat processImg = resizeImageIfTooLarge(img, 1200);

                        // Process with Serial
                        std::cout << "  Running serial version..." << std::endl;
                        cv::Mat serial_result = DarkChannel::dehaze(processImg);
                        DarkChannel::TimingInfo serial_timing = DarkChannel::getLastTimingInfo();

                        // Process with OpenMP
                        std::cout << "  Running OpenMP version..." << std::endl;
                        int num_threads = get_max(1, omp_get_max_threads());
                        cv::Mat openmp_result = DarkChannel::dehazeParallel(processImg, num_threads);
                        DarkChannel::TimingInfo openmp_timing = DarkChannel::getLastTimingInfo();

                        // Store the timings
                        timingPairs.push_back(std::make_pair(serial_timing, openmp_timing));
                        imagePaths.push_back(imgPath);

                        // Process with CUDA if available
                        bool localCudaAvailable = DarkChannel::isCudaAvailable();
                        if (localCudaAvailable) {
                            std::cout << "  Running CUDA version..." << std::endl;
                            try {
                                cv::Mat cuda_result = DarkChannel::dehaze_cuda(processImg);
                                DarkChannel::TimingInfo cuda_timing = DarkChannel::getLastTimingInfo();
                                cudaTimings.push_back(cuda_timing);
                            }
                            catch (const std::exception& e) {
                                std::cerr << "CUDA processing failed: " << e.what() << std::endl;
                                localCudaAvailable = false;
                            }
                        }

                        std::cout << "  Image " << (i + 1) << " complete." << std::endl;
                    }

                    // Export performance data
                    std::string csvFilename = project_path + "performance_data.csv";
                    bool hasCudaData = DarkChannel::isCudaAvailable() && !cudaTimings.empty();

                    exportPerformanceDataToCSV(
                        csvFilename,
                        imagePaths,
                        timingPairs,
                        hasCudaData,
                        hasCudaData ? &cudaTimings : nullptr
                    );

                    std::cout << "\nBenchmarking complete. Data exported to: " << csvFilename << std::endl;
                    std::cout << "You can now import this CSV file into Google Colab for plotting." << std::endl;

                    waitForKeyPress();
                    break;
                }
                
                default:
                    std::cout << "Invalid choice. Please try again." << std::endl;
                    waitForKeyPress();
                }
            }
        }

        cv::destroyAllWindows();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception in main: " << e.what() << std::endl;
        waitForKeyPress();
        return 1;
    }
    catch (...) {
        std::cerr << "Unknown exception in main" << std::endl;
        waitForKeyPress();
        return 1;
    }
}