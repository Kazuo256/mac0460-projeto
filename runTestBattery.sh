#!/bin/bash

for extractor in SURF SIFT
do
  for detector in STAR SIFT SURF ORB #MSER GFTT HARRIS Dense
  do
    for adapter in "" Pyramid
    do
      logname=logs/${1}-${adapter}${detector}-${extractor}
      nice ./$1 $adapter$detector $extractor > $logname.out 2> $logname.err
    done
  done 
done

