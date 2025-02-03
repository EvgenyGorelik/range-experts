project(bev_fusion)
cmake_minimum_required(VERSION 3.17)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

add_definitions(-w)
add_compile_options(-W)
add_compile_options(-std=c++14)
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

# from autoware package
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# path configs
set(CUDAToolkit_ROOT /usr/local/cuda-11)
set(CUDA_TOOLKIT_ROOT_DIR ${CUDAToolkit_ROOT})

set(CMAKE_CXX_FLAGS "-O3 -g -DENABLE_TEXT_BACKEND_STB")
set(CMAKE_CXX_FLAGS_RELEASE "-Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -DENABLE_TEXT_BACKEND_STB")


set(CUDA_NVCC_FLAGS_RELEASE "-Werror=all-warnings -std=-Xcompiler,-Wextra,-Wall,-Wno-deprecated-declarations,-O3,-fPIC -DENABLE_TEXT_BACKEND_STB")
set(CUDA_NVCC_FLAGS_DEBUG   "-O0 -g -DENABLE_TEXT_BACKEND_STB")


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
  # -gencode arch=compute_$ENV{CUDASM},code=compute_$ENV{CUDASM}
  -gencode arch=compute_86,code=compute_86 -std=c++${CMAKE_CXX_STANDARD}
)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(TensorRT_ROOT /usr/local/TensorRT-8.4.2.4)
find_library(NVINFER nvinfer
  REQUIRED
  PATHS ${TensorRT_ROOT}
  PATH_SUFFIXES lib lib64 lib/x64
  NO_DEFAULT_PATH)

find_library(NVINFER_PLUGIN nvinfer_plugin
  REQUIRED
  PATHS ${TensorRT_ROOT}
  PATH_SUFFIXES lib lib64 lib/x64
  NO_DEFAULT_PATH)

find_path(
  TensorRT_INCLUDE_DIR
  NAMES NvInfer.h
  PATHS ${TensorRT_ROOT}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
)

if(NVINFER)
  message(STATUS "TensorRT is available!")
  message(STATUS "NVINFER: ${NVINFER}")
  message(STATUS "NVINFER_PLUGIN: ${NVINFER_PLUGIN}")
  message(STATUS "TensorRT_INCLUDE_DIR: ${TensorRT_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "TensorRT is NOT Available")
endif()



find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(rclpy REQUIRED)
# find_package(autoware_cmake REQUIRED)
# autoware_package()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(yaml-cpp CONFIG REQUIRED)
# get_target_property(LIBRARY_TYPE yaml-cpp::yaml-cpp TYPE)

set(protobuf_BUILD_TESTS OFF)
set(protobuf_BUILD_SHARED_LIBS ON)
set(protobuf_VERBOSE ON)
add_subdirectory(third_party/protobuf/cmake)

include_directories(
  "include"
  ${TensorRT_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${Python_INCLUDE_DIRS}
  ${Protobuf_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_LIBS}
  
  third_party/cuOSD/src
  third_party/3DSparseConvolution/libspconv/include
  third_party/stb

  src/common
  src
)

link_directories(
  ${CUDAToolkit_LIBRARY_DIR}
  ${Python_LIBRARY_DIRS}
  third_party/3DSparseConvolution/libspconv/lib/${arch}
  build
)


file(GLOB_RECURSE CORE_FILES 
  src/bevfusion/*.cu 
  src/bevfusion/*.cpp
  src/common/tensor.cu
  src/common/tensorrt.cpp
)


####################### bevfusion lib #############################
cuda_add_library(${PROJECT_NAME}_cuda_lib SHARED 
  ${CORE_FILES}
)

target_link_libraries(${PROJECT_NAME}_cuda_lib
  CUDA::cudart
  CUDA::cublasLt
  ${NVINFER}
  ${NVINFER_PLUGIN}
  libspconv.so
  libprotobuf
  ${EIGEN3_LIBRARIES}
  ${OpenCV_LIBS}
  ${YAML_CPP_LIBRARIES}
)

########################## libpybev.so ################################
add_library(pybev SHARED
  src/python.cpp
)

target_link_libraries(pybev
  ${PROJECT_NAME}_cuda_lib
)

## node ##
ament_auto_add_library(${PROJECT_NAME}_component SHARED
  src/bevfusion_node.cpp
)


target_link_libraries(${PROJECT_NAME}_component
  ${PROJECT_NAME}_cuda_lib
  ${YAML_CPP_LIBRARIES}
)

rclcpp_components_register_node(${PROJECT_NAME}_component
  PLUGIN "bevfusion::BEVFusionNode"
  EXECUTABLE ${PROJECT_NAME}_node
)

file(GLOB_RECURSE VISUALIZE_FILES 
  third_party/cuOSD/src/*.cpp
  third_party/cuOSD/src/*.cu
  src/common/visualize.cu
)

cuda_add_executable(bevfusion_test
  src/main.cpp
  ${VISUALIZE_FILES}
)
target_link_libraries(bevfusion_test
  ${PROJECT_NAME}_cuda_lib
)

add_executable(bevfusion_param_test
  src/test/test_param.cpp
)
target_link_libraries(bevfusion_param_test
  ${YAML_CPP_LIBRARIES}
)

ament_python_install_package(${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME}_cuda_lib
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS pybev
  DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}"
)

install(PROGRAMS
   bev_fusion/bev_fusion_node.py
   DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})


ament_auto_package(INSTALL_TO_SHARE
  config
  launch
  model
)
