cmake_minimum_required(VERSION 3.18)
project(Dehaze LANGUAGES CXX)

# Set build type to Release for better performance
SET(CMAKE_BUILD_TYPE "Release")

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# Try to auto-detect CUDA installation first
find_package(CUDA QUIET)

if(NOT CUDA_FOUND)
    # If auto-detection fails, try with explicit paths
    message(STATUS "Auto-detection of CUDA failed, trying with explicit paths")
    
    # Windows paths
    if(WIN32)
        set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8")
        if(NOT EXISTS "${CUDA_TOOLKIT_ROOT_DIR}")
            set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3")
        endif()
        if(NOT EXISTS "${CUDA_TOOLKIT_ROOT_DIR}")
            set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2")
        endif()
        if(NOT EXISTS "${CUDA_TOOLKIT_ROOT_DIR}")
            set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
        endif()
    else()
        # Linux/macOS paths
        set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
        if(NOT EXISTS "${CUDA_TOOLKIT_ROOT_DIR}")
            set(CUDA_TOOLKIT_ROOT_DIR "/opt/cuda")
        endif()
    endif()
    
    if(EXISTS "${CUDA_TOOLKIT_ROOT_DIR}")
        message(STATUS "Using CUDA from: ${CUDA_TOOLKIT_ROOT_DIR}")
        set(CUDA_BIN_PATH "${CUDA_TOOLKIT_ROOT_DIR}/bin")
        set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/include")
        
        if(WIN32)
            set(CUDA_LIBRARY_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
        else()
            set(CUDA_LIBRARY_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
        endif()
        
        # Optionally force environment path
        set(ENV{CUDA_PATH} ${CUDA_TOOLKIT_ROOT_DIR})
        
        # Try to find CUDA again with explicit path
        find_package(CUDA)
    else()
        message(WARNING "Could not find CUDA installation. Building without CUDA support.")
    endif()
endif()

# Setup CUDA if found
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DCUDA_ENABLED)
    # Add CUDA as a language after finding it
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
else()
    message(WARNING "CUDA not found. Building without CUDA support.")
endif()

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP found")
else()
    message(WARNING "OpenMP not found. Building without OpenMP support.")
endif()

# Gather all source files
file(GLOB Dehaze_src 
    "*.cpp" 
    "*.h"
)

# Add CUDA sources if CUDA is found
if(CUDA_FOUND)
    file(GLOB Dehaze_cuda_src "*.cu")
    list(APPEND Dehaze_src ${Dehaze_cuda_src})
endif()

# Add executable 
add_executable(Dehaze ${Dehaze_src})

# Set OpenCV directory (adjust to your installation if needed)
if(WIN32)
    set(OpenCV_DIR "C:\\opencv\\build")
else()
    # Try default Linux/macOS locations
    if(EXISTS "/usr/local/share/OpenCV")
        set(OpenCV_DIR "/usr/local/share/OpenCV")
    elseif(EXISTS "/usr/share/OpenCV")
        set(OpenCV_DIR "/usr/share/OpenCV")
    endif()
endif()

# Find OpenCV packages
find_package(OpenCV REQUIRED core imgproc highgui)

# Link libraries
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(Dehaze ${OpenCV_LIBS})
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

# Add CUDA libraries if CUDA is found
if(CUDA_FOUND)
    target_link_libraries(Dehaze ${CUDA_LIBRARIES})
    
    # Set CUDA compilation flags
    set_target_properties(Dehaze PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    # Set architecture if needed
    set(CMAKE_CUDA_ARCHITECTURES "50;60;70;75;80")
endif()

# Create images directory in build directory
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/images)

# Copy image files to build directory if they exist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/images")
    file(GLOB IMG ${CMAKE_CURRENT_SOURCE_DIR}/images/*)
    foreach(img ${IMG})
        get_filename_component(img_name ${img} NAME)
        configure_file(${img} ${CMAKE_CURRENT_BINARY_DIR}/images/${img_name} COPYONLY)
    endforeach()
endif()