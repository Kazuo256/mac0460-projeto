
cmake_minimum_required (VERSION 2.8)
project ( proj )

if (NOT OpenCV_DIR)
  message (FATAL_ERROR "DUDE")
endif ()

include ( ${OpenCV_DIR}/OpenCVConfig.cmake )
find_package ( OpenCV REQUIRED )

include_directories ( src )
add_definitions ( -ansi -pedantic -g -O3 )

add_executable ( bayes_classifier src/main.cxx )
target_link_libraries ( bayes_classifier ${OpenCV_LIBS} )
set_target_properties (
  bayes_classifier
  PROPERTIES
  COMPILE_FLAGS "-DCLASSIFIER=bayes"
)

