#!/usr/bin/env sh

echo Training...
./src/train.py
echo Done.
echo
echo
echo Predicting...
./src/predict.py
echo Benchmark:
./test/benchmark.py

echo
echo Have a nice day
