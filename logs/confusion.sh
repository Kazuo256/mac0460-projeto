#!/bin/bash

echo ================================
for log in bayes_classifier-PyramidSIFT-SIFT.out knn_classifier-PyramidSTAR-SIFT.out rtrees_classifier-PyramidORB-SIFT.out dtree_classifier-PyramidORB-SIFT.out
do
  grep "[0-9]:" $log | awk -f confusion.awk
  echo ================================
done

