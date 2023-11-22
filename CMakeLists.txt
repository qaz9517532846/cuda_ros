cmake_minimum_required(VERSION 3.0.2)
project(cuda_ros)

# add_compile_options(-std=c++11)


find_package(catkin REQUIRED COMPONENTS
  roscpp
)

find_package(CUDA REQUIRED)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES cuda_ros
#  CATKIN_DEPENDS roscpp
#  DEPENDS system_lib
)

# Specify CUDA architecture
# LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_60")

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

# Add executable
cuda_add_executable(cuda_hello_world src/cuda_hello_world.cu)
target_link_libraries(cuda_hello_world ${catkin_LIBRARIES} ${CUDA_LIBRARIES})

cuda_add_executable(cuda_matrix_add src/cuda_matrix_add.cu)
target_link_libraries(cuda_matrix_add ${catkin_LIBRARIES} ${CUDA_LIBRARIES})
