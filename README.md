# Dilated, Residual, Gated CNN

This is a PyTorch implementation of the network presented in Chang et al "Temporal Modeling Using Dilated Convolution and Gating for Voice-Activity-Detection" 2018 [Link to paper](https://ieeexplore.ieee.org/document/8461921)

The network is used for Voice Activity Detection (VAD) in the paper

## Network Architecture 
The core network arcitecture can be seen in the drawing below
![Architecture](/images/network_architecture.png)

The original paper does not state how they do the dimension matching and flattening to the fully connected layer in the end of the network. For the dimension matching, simple 2D convolutions were used. For the flattening, two consecutive 1x1 convolutions were used before flattening to the fully connected layer. 

![](/images/parameters.png)
