#!/bin/bash

for extractor in SURF SIFT
do
  for detector in FAST STAR SIFT SURF ORB MSER GFTT HARRIS Dense
  do
    echo nice ./$1 $2$detector $extractor
  done 
done

