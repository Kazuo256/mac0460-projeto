
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

#ifndef CLASSIFIER
#define CLASSIFIER bayes
#endif

#define CLASSIFIER_CONFIG(classifier) <classifier/config.h>
#include CLASSIFIER_CONFIG(CLASSIFIER)

using std::time_t;
using std::time;

using std::string;
using std::vector;
using std::map;
using std::pair;
using std::make_pair;

using std::ifstream;
using std::ofstream;
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

static vector<Entry>      training_set,
                          test_set;
static map<string,float>  class_labels;
static float              next_label = 0.0;

static string get_dir (const string& str) {
  size_t found = str.find_last_of("/\\");
  string total_dir = str.substr(0,found);
  found = total_dir.find_last_of("/\\");
  return total_dir.substr(found+1);
}

static void load_entryset (const string& filename, vector<Entry>& entryset) {
  ifstream file(filename.c_str(), ios_base::in);
  while (!file.eof()) {
    string img_path;
    getline(file, img_path);
    if (img_path.size() == 0) continue;
    string class_name = get_dir(img_path);
    Mat img = imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);
    entryset.push_back(Entry(img, class_name));
    if (class_labels.count(class_name) == 0)
      class_labels[class_name] = next_label++;
  }
  file.close();
}

static void load_training_set () {
  load_entryset("training.set", training_set);
}

static void load_test_set () {
  load_entryset("test.set", test_set);
}

static time_t last = 0, current = 0;

static void print_done () {
  const static string sep = " ";
  current = time(NULL);
  cout << sep << "Done in" << sep << (current-last) << sep << "seconds.";
  cout << endl;
  last = current;
}

static void write_vocabulary (const string& filename, const Mat& vocabulary) {
  ofstream output(filename.c_str(), ios_base::out);
  output << vocabulary.rows << " " << vocabulary.cols << endl;
  for (int i = 0; i < vocabulary.rows; ++i) {
    for (int j = 0; j < vocabulary.cols; ++j)
      output << vocabulary.at<float>(i,j) << " ";
    output << endl;
  }
  output.close();
}

static void read_vocabulary (const string& filename, Mat& vocabulary) {
  ifstream input(filename.c_str(), ios_base::in);
  size_t rows, cols;
  input >> rows >> cols;
  vocabulary.create(rows, cols, 5);
  for (int i = 0; i < vocabulary.rows; ++i) {
    for (int j = 0; j < vocabulary.cols; ++j)
    input >> vocabulary.at<float>(i,j);
  }
}

template <typename T>
static void train (const Mat& vocabulary, T& classifier, const string& file) {
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

  cout << "Training classifier...";
  cout.flush();
  classifier.train(samples_32f, labels);
  print_done();

  try {
    cout << "Writing classifier to file...";
    cout.flush();
    classifier.save(file.c_str());
    print_done();
  } catch (cv::Exception e) {
    cout << "No support for writing classifiers to file." << endl;
  }

}

int main (int argc, char** argv) {

  if (argc != 3) {
    cout << "Usage:" << endl;
    cout << argv[0] << " <feature_detector> <descriptor_extractor" << endl;
    return 1;
  }

  last = current = time(NULL);

  string detector_name = argv[1],
         extractor_name = argv[2];

  cout << "Loading training set...";
  cout.flush();
  load_training_set();
  print_done();

  Ptr<FeatureDetector>      detector(FeatureDetector::create(detector_name));
  Ptr<DescriptorExtractor>  extractor(DescriptorExtractor::create(extractor_name));
  Mat                       training_descriptors;

  if (detector.empty() || extractor.empty()) {
    cout << "ERROR: Unable to create detector and/or extractor." << endl;
    return 1;
  }

  cout << "Detecting key points and extracting descriptors...";
  cout.flush();
  for (vector<Entry>::iterator it = training_set.begin();
       it != training_set.end(); ++it) {
    detector->detect(it->img, it->keypoints);
    extractor->compute(it->img, it->keypoints, it->descriptors);
    training_descriptors.push_back(it->descriptors);
  }
  print_done();

  Mat vocabulary;
  string vocabulary_file = detector_name+"_"+extractor_name+".vocabulary";

  if (ifstream(vocabulary_file.c_str(), ios_base::in).fail()) {
    cout << "Vocabulary not found." << endl;
    cout << "Generating vocabulary...";
    cout.flush();
    BOWKMeansTrainer trainer(256); //num clusters
    trainer.add(training_descriptors);
    vocabulary = trainer.cluster();
    print_done();
    write_vocabulary(vocabulary_file, vocabulary);
  } else {
    cout << "Loading vocabulary...";
    cout.flush();
    read_vocabulary(vocabulary_file, vocabulary);
    print_done();
  }

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

  CLASSIFIER_T classifier;
  string classifier_file =
    detector_name+"_"+extractor_name+"_"+CLASSIFIER_NAME+".xml";

  if (ifstream(classifier_file.c_str(), ios_base::in).fail()) {
    cout << "Classifier not found." << endl;
    train(vocabulary, classifier, classifier_file);
  }
  else {
    cout << "Loading classifier...";
    cout.flush();
    classifier.load(classifier_file.c_str());
    cout << endl;
  }

  cout << "Loading test set...";
  cout.flush();
  load_test_set();
  print_done();

  size_t count = 0;
  for (vector<Entry>::iterator it = test_set.begin();
       it != test_set.end(); ++it) {
    Mat hist;
    vector<KeyPoint> keypoints;
    detector->detect(it->img,keypoints);
    hist_extractor->compute(it->img, keypoints, hist);
    float answer = predict(classifier, hist);
    float expected = class_labels[it->classname];
    cout << (it-test_set.begin()) << ": ";
    cout << answer << " (should be " << it->classname << ", label ";
    cout << expected << ") ";
    cout << ((answer==expected) ? (++count,"OHYES") : "MAMMAMIA") << endl;
  }
  cout << "Correctly classified: " << count << " (" << test_set.size() << ")";
  cout << endl;

  cout << "BYEBYE" << endl;

  return 0;
}

