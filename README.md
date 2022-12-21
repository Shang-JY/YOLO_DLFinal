# YOLOv1_DLFinal

Hello everyone, this is my repo for YOLOv1 DL-Final-Project. In this project, we started with replacing the backbone network with VGG-16, modified and tested it gradually, and replace the backbone network as resnet50 to train on VOC2007 and VOC2012 datasets. The final model has 26M parameters and achieve over 64% mAP.

In this repository.

The figure folder stores the images of the training process; 

The data folder stores the VOC2007 and VOC2012 dataset; the history folder stores the records of the training process; 

The records folder stores the snapshots of different epochs of the model and parameters of the network. 

The definition and generation mechanism of the residual network are in nets folder, 

We have also written some tools on data preprocessing and loss functions, which are placed in the utils folder.

The code for training and testing is in main.py and train.py.


The settings and accuracies of each model are as follows. We changed, the type of block, layer depth, number of channels, pooling size, we also want to modify the filter size, skip kernel size.

| Name      | # params| Test acc |
|-----------|--------:|:-----------------:|
|VGG16      | 123.51M   | 63.3%|
|ResNet50   |  ~26M   | 64.4%|
|ResNet50v  |  26.73M   | 64.9%|


If you have any questions, please contact me.

Jiayi Shang.

Email: js11640@nyu.edu
