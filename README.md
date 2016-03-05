### start.sh


### download weights
cd darknet/weights
aria2c http://pjreddie.com/media/files/yolo.weights


### change your ~/.bashrc

export CUDA_HOME=/usr/local/cuda-7.5
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}

