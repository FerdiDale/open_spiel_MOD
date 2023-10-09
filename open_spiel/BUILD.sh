#!/bin/bash

BUILD_SHARED_LIB=ON CXX=clang++ cmake -DPython3_EXECUTABLE=$(which python3) -DCMAKE_CXX_COMPILER=${CXX} open_spiel

make -j$(nproc) open_spiel
