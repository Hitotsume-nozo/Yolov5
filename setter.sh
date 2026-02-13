#!/bin/bash
# Fix OpenBLAS threading conflict
export OPENBLAS_CORETYPE=ARMV8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
# Improve CUDA performance on Jetson
export CUDA_LAUNCH_BLOCKING=0

