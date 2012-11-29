
cmake_minimum_required (VERSION 2.8)
project ( proj )

if (NOT OpenCV_DIR)
  message (FATAL_ERROR "DUDE")
endif ()

include ( ${OpenCV_DIR}/OpenCVConfig.cmake )
find_package ( OpenCV REQUIRED )

include_directories ( src )

add_executable ( proj src/main.cxx )
target_link_libraries ( proj ${OpenCV_LIBS} )
set_target_properties ( proj PROPERTIES COMPILE_FLAGS "-ansi -pedantic -g -O3" )

