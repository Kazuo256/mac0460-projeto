
#include <cstdio>
#include <ctime>

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <utility>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/ml/ml.hpp>

using std::time_t;
using std::time;

using std::string;
using std::vector;
using std::map;
using std::pair;
using std::make_pair;

using std::ifstream;
using std::ios_base;
using std::getline;
using std::cout;
using std::endl;

using cv::Ptr;
using cv::Mat;
using cv::KeyPoint;

using cv::FeatureDetector;
using cv::DescriptorExtractor;
using cv::DescriptorMatcher;
using cv::SurfFeatureDetector;
using cv::SurfDescriptorExtractor;
using cv::BOWKMeansTrainer;
using cv::BOWImgDescriptorExtractor;
using cv::FlannBasedMatcher;

using cv::imread;

static void help () {
  printf("\nThis program demonstrates using features2d detector, descriptor extractor and simple matcher\n"
          "Using the SURF desriptor:\n"
          "\n"
          "Usage:\n matcher_simple <image1> <image2>\n");
}

typedef vector<KeyPoint> KeyPoints;

struct Entry {
  Entry (const Mat& dah_img, const string& dah_classname) :
    img(dah_img), classname(dah_classname) {}
  Mat       img;
  string    classname;
  KeyPoints keypoints;
  Mat       descriptors;
  Mat       histogram;
};

static vector<Entry>      training_set;
static map<string,float>  class_labels;
static float              next_label = 0.0;

static string get_dir (const string& str) {
  size_t found = str.find_last_of("/\\");
  string total_dir = str.substr(0,found);
  found = total_dir.find_last_of("/\\");
  return total_dir.substr(found+1);
}

static void load_training_set () {
  ifstream file("training.set", ios_base::in);
  while (!file.eof()) {
    string img_path;
    getline(file, img_path);
    if (img_path.size() == 0) continue;
    string class_name = get_dir(img_path);
    Mat img = imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);
    training_set.push_back(Entry(img, class_name));
    if (class_labels.count(class_name) == 0)
      class_labels[class_name] = next_label++;
  }
}

static time_t last = 0, current = 0;

static void print_done () {
  const static string sep = " ";
  current = time(NULL);
  cout << sep << "Done in" << sep << (current-last) << sep << "seconds.";
  cout << endl;
  last = current;
}

int main (int argc, char** argv) {
  //if (argc != 3) {
  //  help();
  //  return -1;
  //}

  last = current = time(NULL);

  cout << "Loading training set...";
  cout.flush();
  load_training_set();
  print_done();

  Ptr<FeatureDetector>      detector(new SurfFeatureDetector(400));
  Ptr<DescriptorExtractor>  extractor(new SurfDescriptorExtractor);
  Mat                       training_descriptors;

  cout << "Detecting key points and extracting descriptors...";
  cout.flush();
  for (vector<Entry>::iterator it = training_set.begin();
       it != training_set.end(); ++it) {
    detector->detect(it->img, it->keypoints);
    extractor->compute(it->img, it->keypoints, it->descriptors);
    training_descriptors.push_back(it->descriptors);
  }
  print_done();

  cout << "Generating vocabulary...";
  cout.flush();
  BOWKMeansTrainer trainer(256); //num clusters
  trainer.add(training_descriptors);
  Mat vocabulary = trainer.cluster();
  print_done();

  Ptr<DescriptorMatcher>          matcher(new FlannBasedMatcher);
  Ptr<BOWImgDescriptorExtractor>  hist_extractor(
    new BOWImgDescriptorExtractor(extractor, matcher)
  );

  cout << "Setting vocabulary...";
  cout.flush();
  hist_extractor->setVocabulary(vocabulary);
  print_done();

  cout << "Extracting histograms...";
  cout.flush();
  for (vector<Entry>::iterator it = training_set.begin();
       it != training_set.end(); ++it) {
    KeyPoints keypoints;
    detector->detect(it->img, keypoints);
    hist_extractor->compute(it->img, keypoints, it->histogram);
  }
  print_done();

  Mat samples, samples_32f,
      labels;

  cout << "Preparing samples and labels...";
  cout.flush();
  for (vector<Entry>::iterator it = training_set.begin();
       it != training_set.end(); ++it) {
    samples.push_back(it->histogram);
    labels.push_back(Mat(1,1,CV_32F,class_labels[it->classname]));
  }
  samples.convertTo(samples_32f, CV_32F);
  print_done();

  CvNormalBayesClassifier classifier;

  cout << "Training classifier...";
  cout.flush();
  classifier.train(samples_32f, labels);
  print_done();

  cout << "Writing classifier to file...";
  cout.flush();
  classifier.save("SURF_SURF_BAYES.xml");
  print_done();

  cout << "BYEBYE" << endl;

  return 0;
}
