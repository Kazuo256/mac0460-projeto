#!/bin/bash

for classifier in bayes_classifier knn_classifier
do
  for adapter in "" Grid Pyramid
  do
    ./runTestBattery.sh $classifier $adapter &
  done
done

wait

