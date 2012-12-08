#!/bin/bash

if [ ! -d logs ]; then
  mkdir logs
fi

for classifier in bayes_classifier knn_classifier dtree_classifier rtrees_classifier
do
  ./runTestBattery.sh $classifier $adapter &
done

wait

