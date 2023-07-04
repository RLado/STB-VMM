#!/bin/bash
set -e

mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=libtorch/share/cmake/Torch/ ..
cmake --build . --config Release
cd ..