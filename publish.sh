#!/bin/bash

outdir=projeto-gustavo-wilson

if [ ! -d $outdir ]; then
  mkdir $outdir
  mkdir $outdir/test_set
  mkdir $outdir/training_set
fi

contents="src/ CMakeLists.txt logs relatorio.pdf runTest*.sh"
cp -r $contents $outdir

tar -caf $outdir.tar.gz $outdir

rm -rf $outdir

