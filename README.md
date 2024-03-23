# PyTorch Implementation of AlexNet
This is an implementaiton of AlexNet, as introduced in the paper "ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al. ([original paper](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf))

This was the first very successful CNN for image classification that led to breakout of deep learning 'hype', as well as the first successful example of utilizing dropout layers.

# Architecture
The architecture of the network is shown in the image below -
![image.png](https://raw.github.com/suryansh-sinha/AlexNet-CIFAR10/main/images/ArchitectureDiagram.png)

# Prerequisites
- python >= 3.10
- pytorch >= 2.0
  
Everything is included in the google colab file so you need not install anything.

# Dataset
This implementation uses the CIFAR10 dataset released by the University of Toronto. The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

We use the dataset already provided to us by `torchvision.datasets`. The standard Dataset and DataLoader classes are used.

# Training
I have trained the model with data having `batch_size=32`, `learning_rate=0.001` and have used the `Adam` optimizer. This was done to improve the performance of the model since I don't powerful hardware to train it locally on my machine.

The model has an accuracy of 76% on the test dataset when trained on 50 epochs.
