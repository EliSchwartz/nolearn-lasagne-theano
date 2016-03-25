#!/bin/bash

source env create
source env update
source activate nolearn-lasagne-theano

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH
export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,cuda.root=/usr/local/cuda-7.5
