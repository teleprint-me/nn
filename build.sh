#!/usr/bin/env bash

# Create or clean the build directory
build_dir="build"
if [ -d "$build_dir" ]; then
  echo "Cleaning existing build directory..."
  rm -r "$build_dir" || exit 1
fi
cmake -B build -DCMAKE_BUILD_TYPE=Debug || exit 1

# Run CMake and build the project
if cmake --build build -j "$(nproc)"; then
  echo "Build successful!"
  exit 0
else
  echo "Build failed."
  exit 1
fi
