# Install Caffe2 for video models
installing PyTorch through conda does not install the necessary video libraries.
You must install PyTorch/Caffe2 libraries from source to use video models

*tested for CUDA 10.0, cudnn 7.4.2.12-cuda-10.0, openmpi 3.1.0*

# load necessary CUDA/cudnn libraries
module load cuda/10.0
module load cudnn/7.4.2.12-cuda-10.0

# load MPI
module load openmpi

# add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<cuda lib dir>

# get PyTorch from source
git clone --recursive https://github.com/pytorch/pytorch

# make sure to have opencv
conda install -c conda-forge opencv

# ffmpeg
conda install -c conda-forge ffmpeg 


# install python-lmdb
conda install -c conda-forge python-lmdb

cd <PyTorch dir>
USE_FFMPEG=1 USE_OPENCV=1 USE_LEVELDB=0 USE_LMDB=1 python setup.py install

# test installation
python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

# VMZ/facebook PYTHONPATH
export PYTHONPATH=$PYTHONPATH:<VMZ lib dir>
