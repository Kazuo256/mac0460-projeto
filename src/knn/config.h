
#ifndef MAC0460_KNN_CONFIG_H_
#define MAC0460_KNN_CONFIG_H_

#include <opencv2/core/core.hpp>

#define CLASSIFIER_T    CvKNearest
#define CLASSIFIER_NAME "KNN"

inline float predict (CLASSIFIER_T& classifier, const cv::Mat& samples) {
  return classifier.find_nearest(samples, 1);
}

#endif

