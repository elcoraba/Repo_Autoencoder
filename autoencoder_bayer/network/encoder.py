import numpy as np
import torch.nn as nn
import torch

from network.residualblock import ResidualBlock 

class Encoder(nn.Module):
    def __init__(self, kernel_size, filters, dilations, downsamples, causal, nInput = 2):
        super(Encoder, self).__init__()
        #TODO? ich brauche den ganzen self. Quatsch doch garnicht? Muss ich von wo anders darauf zugreifen?
        self.kernel_size = kernel_size #3
        self.filters = filters # ENCODER_FILTERS = [256, 256, 256, 256]
        self.dilations = dilations #  ENCODER_DILATIONS = [(1, 1), (2, 4), (8, 16), (32, 64)]
        self.downsamples = downsamples # ENCODER_DOWNSAMPLE = [0, 0, 0, 0]
        self.causal = causal #they imported it with args TODO? ist Encoder nicht immer non-causal? einfach auf false setzen?
        self.nInput = nInput

        self.out_dim1 = filters[1]  #GazeMAE: out_dim   Vir: out_dim1
        self.out_dim3 = filters[3]  #GazeMAE: out_dim2  Vir: out_dim2

        self.enc = nn.Sequential(
            # nInput, nHidden, nOutput, dilations, kernel_size, causal, downsample = 0, no_skip_connections = False
            ResidualBlock(self.nInput, self.filters[0], self.filters[0], self.dilations[0], self.kernel_size, self.causal, self.downsamples[0]),
            ResidualBlock(self.filters[0], self.filters[1], self.filters[1], self.dilations[1], self.kernel_size, self.causal, self.downsamples[1]),
            ResidualBlock(self.filters[1], self.filters[2], self.filters[2], self.dilations[2], self.kernel_size, self.causal, self.downsamples[2]),
            ResidualBlock(self.filters[2], self.filters[3], self.filters[3], self.dilations[3], self.kernel_size, self.causal, self.downsamples[3])
        )

    def forward(self, x):
        micro_out = self.enc[1](self.enc[0](x))
        macro_out = self.enc[3](self.enc[2](micro_out))

        return micro_out.mean(-1), macro_out.mean(-1) # GAP - Global Average Pooling

