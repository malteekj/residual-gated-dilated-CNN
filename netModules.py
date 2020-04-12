import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import time

'''
Main building block of network: Residual, dilated, gated CNN 
'''
class RCNNBlock(nn.Module):
  '''
  Args:
      input_channels:           Number of input channels 
      dilation:                 Size of the dilation
      kernel_size:              Size of the convolutional kernel
      padding:                  Zero padding of the input. If None, the size of the input will be kept 
      bias:                     If bias should be used 
      dropout:                  Dropout rate to use 
  '''

  def __init__(self,input_channels,dilation,kernel_size,
               padding=None, bias=True, dropout=0.05):
    super(RCNNBlock,self).__init__()

    # weight inits
    def init_weights(m):
          if type(m) == nn.Conv2d:
              torch.nn.init.xavier_normal_(m.weight)
              m.bias.data.fill_(0.01)
    self.dropout = dropout

    if padding is None:
      padding = (kernel_size - 1)//2 * dilation
    
    # Convolutional layer
    self.convLayer = nn.Conv2d(in_channels=input_channels,
                  out_channels=input_channels*2, 
                  dilation=dilation, kernel_size=kernel_size, 
                  padding=padding, bias=bias)
    
    # For batchnorm
    self.BN2d_a = nn.BatchNorm2d(input_channels,eps=1e-05, momentum=0.1, affine=True)
    self.BN2d_b = nn.BatchNorm2d(input_channels,eps=1e-05, momentum=0.1, affine=True)
    
    # Initialize weights
    self.convLayer.apply(init_weights)

  def forward(self,x):
    res = x.clone()
    x = F.dropout(x, p=self.dropout, training=self.training)
    x = self.convLayer(x)
    a, b = torch.chunk(x, 2, dim=1)
    a = self.BN2d_a(a)
    b = self.BN2d_b(b)
    x = torch.tanh(a) * torch.sigmoid(b)
    return x + res

'''
Dilated convolutional residual network
'''
class Net(nn.Module):
    '''
    Complete netork from the paper Chang et al. "TEMPORAL MODELING USING DILATED CONVOLUTION AND GATING FOR VOICE-ACTIVITY-DETECTION" (2018)

    Args:
        cin_channels:           Number of input channels 
        gate_channels:          Number of channels in the gates
        num_classes:            Number of classes to predict
        num_layers:             Number of convolutional layers (RCNNBlocks)
        num_stacks:             Number of dilations before repeat of the dilation pattern 
        kernel_size:            Size of the convolutional kernel
        dropout:                Global dropout rate used in the network
        input_size:             Size of the input used for the fully connected layers

    Shape:
        Input:                  Expected to be an MxN matrix of mel spectrogram
        Output:                 Probabilities of the K classes 
    '''

    def __init__(self, cin_channels, gate_channels, num_classes, num_layers, 
                 num_stacks, kernel_size, dropout, input_size):
        super(Net, self).__init__()
        
        self.num_classes = num_classes
        self.cin_channels = cin_channels
        self.gate_channels = gate_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.num_stacks = num_stacks
        self.num_layers = num_layers
        self.input_size = input_size

        # input block
        self.convBlock0 = nn.Conv2d(in_channels=cin_channels,
                                out_channels=gate_channels, 
                                kernel_size=kernel_size, 
                                padding=(kernel_size-1)//2)
        self.bn_input = nn.BatchNorm2d(gate_channels,eps=1e-05, momentum=0.1, affine=True)
        
        # Main convolutional layers
        assert num_layers % num_stacks == 0
        layers_per_stack = num_layers // num_stacks

        self.convLayers = nn.ModuleList()
        for layer in range(num_layers):
          dilation = 2**(layer % layers_per_stack)
          conv =  RCNNBlock(input_channels=gate_channels, 
                            dilation=dilation,kernel_size=kernel_size,dropout=dropout)
          self.convLayers.append(conv)

        # Dim reduction before fully connected
        self.conv1x1_1 = nn.Conv2d(in_channels=gate_channels,
                                out_channels=8,
                                kernel_size = 1,
                                padding=0)
        self.conv1x1_2 = nn.Conv2d(in_channels=8,
                                out_channels=2,
                                kernel_size = 1,
                                padding=0)
        
        self.bn0_reduction = nn.BatchNorm2d(gate_channels,eps=1e-05, momentum=0.1, affine=True)
        self.bn1_reduction = nn.BatchNorm2d(8,eps=1e-05, momentum=0.1, affine=True)
                  
        self.num_out = self.input_size[0]*self.input_size[1]*2
        
        # Fully connected
        self.fc1 = nn.Linear(self.num_out, 1024)
        self.fc2 = nn.Linear(1024,256)
        self.fc3 = nn.Linear(256,num_classes)
        
        self.bn0 = nn.BatchNorm1d(self.num_out)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc_do = nn.Dropout(p=dropout)
        
        # Initialize fully connected
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        x = self.convBlock0(x)
        x = self.bn_input(x)
        x = F.relu(x)
        
        # Convolutional blocks
        x_res = x.clone()
        for layer in self.convLayers:
          x = layer(x)
          x_res = x_res + x
        x_res = x_res * np.sqrt(1/len(self.convLayers))

        # Dimension reduction before fully connected
        x = self.bn0_reduction(x_res)
        x = F.relu(x)
        x = self.conv1x1_1(x)
        x = self.bn1_reduction(x)
        x = F.relu(x)
        x = self.conv1x1_2(x)
        
        # Fully connected
        x = x.view(-1,self.num_flat_features(x))
        x = self.bn0(x)
        x = F.relu(x)
        x = self.fc_do(x)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc_do(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc_do(x)
        
        x = self.fc3(x)
        
        return F.softmax(x, dim = 1) 
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1 # initialize
        for s in size:
            num_features *= s
        return num_features
