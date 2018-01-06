In this video, I will show you how to install `CUDA` + `CUDNN` + `ANACONDA` + `TENSORFLOW`.

1. Download **CUDA**:
   1. Link: https://developer.nvidia.com/cuda-downloads
   2. Use CUDA 8 for Tensorflow 1.4 / CUDA 9 for Tensorflow 1.5+
   3. Tensorflow 1.5 is only in beta so I will use CUDA 8 now. Please check the latest Tensorflow version. You may need CUDA 9 at the time of watching this video. Now, I will download CUDA 8.
2. Download **CUDNN**:  
   1. Link: https://developer.nvidia.com/cudnn
   2. Use CUDNN 6 for Tensorflow 1.4 / CUDNN 9 for Tensorflow 1.5+
3. Download **Anaconda**
   1. Link: https://conda.io/miniconda.html
   2. You can choose Python 2.7 based or Python 3.6 based Miniconda. 
4. Install CUDA
5. Install CUDNN
   1. Link: http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installwindows
6. Install Anaconda
7. Create an environment from an `environment.yml` file
   1. Link: https://github.com/quanhua92/deeplearning.vn/tree/master/series/cai_dat_phan_mem
   2. There are 4 `.yml` files in this folder:
      1. `env2.yml`: `Python 2.7` + `Tensorflow CPU-only` 
      2. `env2-gpu.yml`: `Python 2.7` + `Tensorflow GPU`  
      3. `env3.yml`: `Python 3.5` + `Tensorflow CPU-only` 
      4. `env3-gpu.yml`: `Python 3.5` + `Tensorflow GPU`  
   3. For Windows with GPU-enabled card, we can only use Python 3.5 (`env3-gpu.yml`).
   4. `conda env create -f env3-gpu.yml` 
8. Test the installation
   1. Open Anaconda console
   2. `activate env3-gpu`
   3. `python`
   4. `import tensorflow as tf`
   5. `sess = tf.Session()` 

Tensorflow can recognize my GeForce GTX 1070 with total Memory of 8.00 GB and 6.68GB free memory.

Great!