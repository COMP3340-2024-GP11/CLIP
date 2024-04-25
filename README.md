# COMP3340 Group 10 - CLIP + Data Augmentation with Segmentation Image #

## Contact ##

- This repository contains code for CLIP Zero-shot Classification on Oxford 17 Dataset and data augmentation with segmentation image
- For any question and enquiry, please feel free to reach out to Zhong Zhiyi(zhongzy@connect.hku.hk)

## Overview

**Prerequisite for Reproduction**

1. [Set up conda environment](#env_setup)
2. [Download data and put them under the correct folder](#downloads)
3. [Run the commands to produce the results](#cmd_repro)

**Software, Hardware & System Requirements**

- Software (Set up environment as [following](#env_setup))
  - python==3.8.18
  - mmcv==2.1.0  
  - mmengine==0.10.3
  - mmpretrain==1.2.0 
  - numpy==1.24.4
  - opencv-python==4.9.0.80
- Hardware (HKU GPU Farm)
  - Experiments are conducted on one NVIDIA GeForce RTX 2080 Ti 
- System
  - Linux

## Environment setup <a id="env_setup"/>

### Basic Setup 

**Step 1. Create virtual environment using anaconda**

\# Please make sure that you are create a virtual env with **python version 3.8**

```
conda create -n mmpretrain python=3.8 -y
conda activate mmpretrain
```

**Step 2. Install Pytorch from wheel**

\# For those interested, please see https://download.pytorch.org/whl/torch/

```
wget https://download.pytorch.org/whl/cu111/torch-1.10.1%2Bcu111-cp38-cp38-linux_x86_64.whl#sha256=3d35d58cadb5abbfa25a474a33598a6bdc168c4306c3c20968159e6f3a4a2e46
pip install torch-1.10.1+cu111-cp38-cp38-linux_x86_64.whl
pip install numpy --upgrade
```

##Please double check that you install the GPU version of pytorch using the following command

```
python
>>> import torch
>>> torch.__version__
```

the correct output should be

```
'1.10.1+cu111'
```

**Step 3 Install cudatoolkit via conda-forge channel**

\# You must be on the GPU compute node to install cudatoolkit and mmcv since GCC compiler and CUDA drivers only available on GPU computing nodes

```
gpu-interactive
conda activate mmpretrain
conda install cudatoolkit=11.1 -c pytorch -c conda-forge -y
```

**Step 4 Install torchvision, openmim package using pip & use openmim to install mmpretrain from source**

\# make sure you are on GPU compute node

```
gpu-interactive
```

\# install packages

```
conda activate mmpretrain
pip install torchvision==0.11.2
git clone https://github.com/COMP3340-2024-GP11/CLIP.git
cd CLIP
pip install -U openmim && mim install -e .
```

**Step 5 An additional step to use CLIP**

```
pip install -q --extra-index-url https://download.pytorch.org/whl/cpu gradio "openvino>=2023.1.0" "transformers[torch]>=4.30" "datasets" "nncf>=2.6.0"
```

#It does not matter if an error occurs like "ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts..."

## Download data and manipulate data<a id="downloads"/>

```
#download flowers data
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz
tar zxvf 17flowers.tgz
mv 17flowers data/flower
cd data

#copy split.py generate_meta.py to flower directory to split the data
cp split.py flower
cp generate_meta.py flower
cd flower
python split.py
mkdir meta
python generate_meta.py
cd ..
```

## Commands to reproduce results<a id="cmd_repro"/>

### Part A: Zero-shot Classification with CLIP

First, please make sure you are in the main directory CLIP instead of CLIP/data

Then, you can directly run the program clip.py by

```
python clip.py
```

The program will print the test accuracy of clip from class 0 to class 16 and finally the general accuracy (the testing of every class might take 4 to 5 minutes):

```
class 0: 1.0
class 1: 0.9875
...
```

Alternatively, you can open any IDE that supports Python notebook and open the file **clip.ipynb** to see the visualization demo of a CLIP inference with or without prompts. If you would like to view the inference result of different images, just change the file name in the correct file name format as specified in the file.

**Note: the result might have slight changes in different running**

### Part B: Data augmentation using segmentation image

First, download the groundtruth segmentation image to the data folder

```
##download segmentation image data and please make sure you are in the main directory
cd data
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/17/trimaps.tgz  
tar zxvf trimaps.tgz
```

Then, you can use **generate_segment.py** to augment data with segmentation masks and add perturbations in the background. You can operate as follows:

```
python generate_segment.py
```

Split the data again

```
cp split.py flowers
cp generate_meta.py flowers
cd flowers
python split.py
mkdir meta
python generate_meta.py
cd ..
```

After this step, all images in the folder 'flowers' have been augmented and can be used for **training**

```
cd .. #make sure you are in the main directory
python tools/train.py \
configs/resnet/resnet18_flowers_bs128.py \
--work-dir output/resnet18_flowers_bs128
```

After training, you use the following command to test the model (as I have modified configs/__base__/datasets/flowers.py, the testing will still be done on the original data)

```
python tools/test.py configs/tinyvit/resnet/resnet18_flowers_bs128.py output/resnet18_flowers_bs128/epoch_200.pth --out output/resnet18_flowers_bs128/test.pkl
```

-- --

After this, all have been done. Thanks for your patience!







