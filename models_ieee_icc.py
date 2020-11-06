"""
# Author
Jakob Krzyston (jakobk@gatech.edu)

# Purpose
Build architecture for I/Q modulation classification as seen in Krzyston et al. 2020
"""

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


##### LINEAR COMBINATION FOR COMPLEX CONVOLUTION #####

class LC(nn.Module):
    def __init__(self):
        super(LC, self).__init__()
        #this matrix adds the first and third columns of the output of Conv2d
    def forward(self, x):
        i = x[:,:,0:1,:]-x[:,:,2:3,:]
        q = x[:,:,1:2,:]
        return torch.cat([i,q],dim=2)
    

##### CLASSIFIER FROM KRZYSTON ET AL. 2020 #####

class Complex(nn.Module):   
    def __init__(self,
        n_classes: int = 11
        ):
        super(Complex, self).__init__()

        # define the dropout layer
        self.dropout  = nn.Dropout(p = 0.5)
        
        # convolutional layers w/ weight initialization
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(2,3), stride=1, padding = (1,1), bias = True)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(256, 80, kernel_size=(2,3), stride=1, padding = (0,1), bias = True)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        
        # dense layers w/ weight initialization
        self.dense1 = nn.Linear(80*128, 256, bias =True)
        torch.nn.init.kaiming_normal_(self.dense1.weight, nonlinearity='relu')
        self.dense2 = nn.Linear(256,n_classes, bias = True)
        torch.nn.init.kaiming_normal_(self.dense2.weight, nonlinearity='sigmoid')
        
    # Defining the forward pass    
    def forward(self, x):
        x = self.conv1(x)
        x = LC.forward(self,x)
        x = F.relu(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x