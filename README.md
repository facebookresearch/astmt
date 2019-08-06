# Multi-task learning
Multi-task learning for computer vision, with task-specific modulation and adversarial training.
Internship project of [Kevis-Kokitsi Maninis](http://www.vision.ee.ethz.ch/~kmaninis/), together with Iasonas Kokkinos and Ilija Radosavovic. This repo will be updated with a technical report.

###  Installation / Setup:
This repo uses synchronized batchnorm layers to train on multiple GPUs with large effective batch size. The [Synchronized Batchnorm](http://hangzh.com/PyTorch-Encoding/) package is third-party, and needs installation
of Pytorch from source.
To avoid compatibility issues, please install using the following:

1. Download and install [Anaconda Python 3.6](https://www.anaconda.com/download/). Alternatives are ofcourse possible.
2. Create an environment and install dependencies.
    ```
    conda create -n multitask python=3.6
    source activate multitask
    
    # Needed to install Pytorch
    conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing requests future
    conda install -c mingfeima mkldnn
    conda install -c pytorch magma-cuda90
    ```
3. Download and install CUDA and cuDNN (The code was tested with CUDA 9.0 and cuDNN 7.0).
For FAIR cluster the `module load` commands can be also used.

4. Install PyTorch from source, but revert to a specific commit first to avoid incompatibilities.
    ```
    # Clone the repo
    git clone --recursive https://github.com/pytorch/pytorch.git
    cd pytorch
    
    # Revert to specific commit
    git reset --hard 13de6e8
    git submodule update --init
    
    # Only for Volta GPUs
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;6.1;7.0"

    # Install pytorch - be patient please.
    python setup.py install
    cd ..

    ```
5. Install torchvision and ninja.
    ```
    # Install torchvision from source
    git clone https://github.com/pytorch/vision.git
    cd vision
    python setup.py install
    cd ..
    
    # Get ninja
    wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
    unzip ninja-linux.zip
    mv ninja /path/to/anaconda/bin
    ```

6. Install Synchronized Batchnorm package (my local copy, with the hacks needed to make it work). Currently syncbnorm works only with 2+ GPUs, meaning that for single GPU usage you have to switch to `nn.BatchNorm2d`.
This is done automatically inside the scripts for this repo.
    ```
    # Install pytorch encoding
    git clone https://github.com/kmaninis/pytorch-encoding.git
    cd pytorch-encoding
    python setup.py install
    cd ..
    ```
    
7. Install (final - promise) additional dependencies.
    ```
    conda install scikit-learn scikit-image pillow cython
    pip install  graphviz opencv-python easydict pycocotools
    
    # For tensorboard, tensorflow is also needed for the moment.
    pip install tensorboard tensorboardx tensorflow
    ```
    
8. Clone the multi-task repo.
    ```
    git clone git@github.com:fairinternal/multi-task.git
    ```

9. Add the multi-task package directory, easiest if you initialize it into your ~/.bashrc.
    ```
    echo "export PYTHONPATH=$PYTHONPATH:/path/to/this/repo/" >> ~/.bashrc
    ```
    
10. Move `mypath.py` under `util/`. Complete the `/path/to/something/` paths. In theory, this is the only part of the code that is user-dependent.
The data used for the project can be found under `/private/home/kmaninis/Workspace/datasets`:
- `PASCAL` for PASCAL (SBD, VOC12, PASCAL in detail, and PASCAL-Context).
- `FSV` for GTA (Free Supervision from Video games - CVPR 2018).
- `NYUD_MT` for NYUD for Multi-tasking.



### What is under fblib?
Main folder `fblib/` contains the basic code needed to train and test:
 - `dataloaders/`: The data loading class that loads images and ground-truth(s).
 - `layers/`: All custom layers, including custom losses go here.
 - `networks/`: VGG, ResNet, Deeplab, Hourglass etc. go here.
 - `evaluation/`: Evaluation scripts for multiple tasks.
 - `external/`: External libraries such as details_api go here.
 

### Experiments

Inside `experiments/` there is code for the main experiments performed in this internship. Under `experiments/classification/` there are minimal multi-tasking 
examples for CIFAR100 and MNIST.

Under `experiments/dense_predict/` main experiments using a single decoder are `*_single_dec_se`.
In order to run multi-tasking on PASCAL (5 tasks) using a single decoder, with SE modules and adversarial loss, try:
```
cd pascal_single_dec_se
python train.py --active_tasks 1 1 1 1 1 --arch se_res101 --pretr imagenet -lr 0.001 --trBatch 8 --epochs 60 --cls atrous-v3 --stride 16 --trNorm True --overfit False  --dec_w 64 --dscr fconv --dscrk 3 --dscrd 2  --semt True --sedec True --newop True 
```

This will train SE-ResNet-101 with Atrous-V3 decoder on 5 tasks. Let's go through the arguments:
- `--active_tasks`: [edge detection, semantic segmentation, human part segmentation, normals, saliency]
- `--arch se_res101`: Use [Squeeze and Excitation](https://arxiv.org/abs/1709.01507) ResNet 101 as backbone.
- `--pretr imagenet`: Use model pre-trained on ImageNet
- `-lr 0.001 --trBatch 8 --epochs 60`: learning rate, training batch, and number of epochs
- `--cls atrous-v3`: Use Atrous-v3 (ASPP) classifier
- `--stride 16`: Total stride of backbone. Choose between 8 and 16, although 16 saves a lot of memory!
- `--trNorm True`: Train the batchnorm layers.
- `--dec_w 64`: width of decoder
- `--overfit False `: Switch to True to overfit into the first samples of the training data
- `--dscr fconv`: Use fully convolutional discriminator
- `--dscrk 3`: Use kernel size 3 for discriminator
- `--dscrd 2`: Use depth of 2 for discriminator.
- `--semt True`: Use squeeze and excitation per task for encoder
- `--sedec True`: Use squeeze and excitation per task for decoder
- `--newop True`: Current implementation of optimizers updates the weights of all layers of the network, even if they are not used during forward pass.
This can happen when momentum is used, for example. Setting `--newop` to True solves this issue.


#### How to run gridsearch on the FAIR cluster
1. Specify your parameters in `experiments/dense_predict/pascal_single_dec_se/pascal_single_dec_se.json`
2. `cd experiments/dense_predict` 
3. `chmod +x submit_job.sh` 
3. `python stool.py run --partition uninterrupted --sweep pascal_single_dec_se`
4. Make sure to activate your environment in `~/.bashrc`, or feel free to modify the bash script.


#### Use Tensorboard
- `cd exp_save_dir`
- `tensorboard --logdir . --port XXXX` .If you are using port forwarding to your local machine, access through `localhost:XXXX`.


#### Evaluation
Evaluation scripts are run at the end of each experiments, and the results are dumped into the experiment folder.
The evaluation part of boundary detection is disabled for FAIR cluster, since it needs MATLAB.

Briefly, the following metrics has been implemented:
- `edge`: F-measures per dataset (odsF), and per Instance (oisF), and Average Precision (AP) (using the [seism](https://github.com/jponttuset/seism) software) - disabled for FAIR cluster.
- `semseg`: mean Intersection over Union (mIoU) per class.
- `human_part`: mean Intersection over Union (mIoU) per class.
- `normals`: Difference in predicted and ground-truth angles (mean, median, RMSE, < 11.25, < 22.5, < 30).
- `sal`: mean Intersection over Union (mIoU).
- `albedo`: Mean Squared Error (MSE).
- `depth`: Mean Squared Error (MSE).

Enjoy :)


 
