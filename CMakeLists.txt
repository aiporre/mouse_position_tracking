cmake_minimum_required(VERSION 3.10)
project(mouse_position_tracking)

set(CMAKE_CXX_STANDARD 14)

#find_package( OpenCV REQUIRED )

# add_executable(mouse_position_tracking main.cpp)

set(SOURCE_FILES main.cpp darkchannel/include/darkchannel.h darkchannel/darkchannel.cpp darkchannel/include/darkchannel.h)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
#include_directories(${OpenCV_INCLUDE_DIRS})
#target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )