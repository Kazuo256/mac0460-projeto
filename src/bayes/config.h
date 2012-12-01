
#ifndef MAC0460_BAYES_CONFIG_H_
#define MAC0460_BAYES_CONFIG_H_

#include <opencv2/core/core.hpp>

#define CLASSIFIER_T    CvNormalBayesClassifier
#define CLASSIFIER_NAME "BAYES"

inline float predict (CLASSIFIER_T& classifier, const cv::Mat& samples) {
  return classifier.predict(samples);
}

#endif

