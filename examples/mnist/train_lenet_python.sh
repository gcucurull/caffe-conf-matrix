#!/usr/bin/env sh

export PYTHONPATH=$PWD/python:$PWD/examples/mnist

./build/tools/caffe train --solver=examples/mnist/lenet_solver_python.prototxt
