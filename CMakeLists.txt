
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

add_executable ( knn_classifier src/main.cxx )
target_link_libraries ( knn_classifier ${OpenCV_LIBS} )
set_target_properties (
  knn_classifier
  PROPERTIES
  COMPILE_FLAGS "-DCLASSIFIER=knn"
)

add_executable ( dtree_classifier src/main.cxx )
target_link_libraries ( dtree_classifier ${OpenCV_LIBS} )
set_target_properties (
  dtree_classifier
  PROPERTIES
  COMPILE_FLAGS "-DCLASSIFIER=dtree"
)

add_executable ( rtrees_classifier src/main.cxx )
target_link_libraries ( rtrees_classifier ${OpenCV_LIBS} )
set_target_properties (
  rtrees_classifier
  PROPERTIES
  COMPILE_FLAGS "-DCLASSIFIER=rtrees"
)

