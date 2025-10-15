#!/bin/sh
export PYTHONPATH=./:$PATH
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH$
export TORCH_CUDA_ARCH_LIST="8.0" # CUDA11.X，对应的算力为8.0

# conda activate spinquant
# conda activate spinquant_minicpm_numpy1.24.4
conda activate Hunyuan-taylor
proxy_on