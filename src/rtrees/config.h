
#ifndef MAC0460_RTREES_CONFIG_H_
#define MAC0460_RTREES_CONFIG_H_

#include <opencv2/core/core.hpp>

#define CLASSIFIER_T    CvRTrees
#define CLASSIFIER_NAME "RTREES"

inline float predict (CLASSIFIER_T& classifier, const cv::Mat& samples) {
  return classifier.predict(samples);
}

#define CUSTOM_TRAINER

inline void train (CLASSIFIER_T& classifier, const cv::Mat& samples,
                   const cv::Mat& labels) {
  classifier.train(samples, CV_ROW_SAMPLE, labels);
}

#endif

