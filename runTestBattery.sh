#!/bin/bash

for extractor in SURF SIFT
do
  for detector in FAST STAR SIFT SURF ORB MSER GFTT HARRIS Dense
  do
    logname=logs/${1}-${2}${detector}-${extractor}
    echo "nice ./$1 $2$detector $extractor > $logname.out 2> $logname.err"
  done 
done

