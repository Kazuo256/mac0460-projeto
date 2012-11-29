
#include <cstdio>

#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using std::string;
using std::vector;
using std::pair;
using std::make_pair;

using std::ifstream;
using std::ios_base;
using std::cout;
using std::endl;

using cv::Mat;
using cv::KeyPoint;

using cv::SurfFeatureDetector;
using cv::SurfDescriptorExtractor;
using cv::BFMatcher;
using cv::DMatch;

using cv::imread;
using cv::namedWindow;
using cv::drawMatches;
using cv::waitKey;

using cv::NORM_L2;

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
};

static vector<Entry>      training_set;
static vector<KeyPoints>  entry_keypoints;

static string get_dir (const string& str) {
  size_t found = str.find_last_of("/\\");
  string total_dir = str.substr(0,found);
  found = total_dir.find_last_of("/\\");
  return total_dir.substr(found+1);
}

static void load_training_set () {
  ifstream file("training.set", ios_base::in);
  while (file.good()) {
    string img_path;
    file >> img_path;
    string class_name = get_dir(img_path);
    Mat img = imread(img_path, CV_LOAD_IMAGE_GRAYSCALE);
    training_set.push_back(Entry(img, class_name));
  }
}

int main (int argc, char** argv) {
  //if (argc != 3) {
  //  help();
  //  return -1;
  //}

  cout << "Loading training set..." << endl;
  load_training_set();

  SurfFeatureDetector     detector(400);
  SurfDescriptorExtractor extractor;

  cout << "Detecting key points and extracting descriptors..." << endl;
  for (vector<Entry>::iterator it = training_set.begin();
       it != training_set.end(); ++it) {
    detector.detect(it->img, it->keypoints);
    extractor.compute(it->img, it->keypoints, it->descriptors);
  }

  cout << "BYEBYE" << endl;

  if (false) {

    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    if (img1.empty() || img2.empty()) {
      printf("Can't read one of the images\n");
      return -1;
    }

    // detecting keypoints
    SurfFeatureDetector detector(400);
    vector<KeyPoint> keypoints1, keypoints2;
    detector.detect(img1, keypoints1);
    detector.detect(img2, keypoints2);

    // computing descriptors
    SurfDescriptorExtractor extractor;
    Mat descriptors1, descriptors2;
    extractor.compute(img1, keypoints1, descriptors1);
    extractor.compute(img2, keypoints2, descriptors2);

    // matching descriptors
    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // drawing the results
    namedWindow("matches", 1);
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
    imshow("matches", img_matches);
    waitKey(0);

  }

  return 0;
}
