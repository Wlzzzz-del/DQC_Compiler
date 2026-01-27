import torch
from torch import nn
import math


#learning agent NNs used by the simulator for distributed quantum computer simulator defined here

class customizedNN(nn.Module):  
    
    def __init__(self, input_dim, output_dim, hidden_layers, device):
        super(customizedNN,self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_layers
        self.device = device
        
        
        print("####################################")
        print("Device for NN: ", self.device)
        print("####################################")
        
        self.layers = nn.ModuleList()
        
        
        for layer in range(len(self.hidden_dims)+1):
            if layer == 0:
                self.layers.append(nn.Linear(self.input_dim, self.hidden_dims[layer]))
                self.layers.append(nn.ReLU())
            elif layer == len(self.hidden_dims):
                self.layers.append(nn.Linear(self.hidden_dims[layer-1], self.output_dim))
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Linear(self.hidden_dims[layer-1], self.hidden_dims[layer]))
                self.layers.append(nn.ReLU())
        
        #self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(0.1)
        
    def forward(self,x, mask):
        
        for layer in self.layers:
            x = layer(x)
            
        x = x * mask
        return x


class customizedNN_policyGrad(nn.Module):  
    
    def __init__(self, input_dim, output_dim, hidden_layers, device):
        super(customizedNN_policyGrad,self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_layers
        self.device = device
        
        
        print("####################################")
        print("Device for NN in policy grad: ", self.device)
        print("####################################")
        
        self.layers = nn.ModuleList()
        
        
        for layer in range(len(self.hidden_dims)+1):
            if layer == 0:
                self.layers.append(nn.Linear(self.input_dim, self.hidden_dims[layer]))
                self.layers.append(nn.ReLU())
            elif layer == len(self.hidden_dims):
                self.layers.append(nn.Linear(self.hidden_dims[layer-1], self.output_dim))
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Linear(self.hidden_dims[layer-1], self.hidden_dims[layer]))
                self.layers.append(nn.ReLU())
        
        self.softmax = nn.Softmax()
        #self.dropout = nn.Dropout(0.1)
        
    def forward(self,x, mask):
        
        for layer in self.layers:
            x = layer(x)
            
        #x = x * mask   
        x = self.softmax(x)   
        if torch.count_nonzero(x) == 0:
            #actor_output[0,0] = 1
            print("minor force fix, action will be 0/stop since NN generated all-zeros")    
        return x