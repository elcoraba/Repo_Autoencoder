import numpy as np
import torch.nn as nn # so that it shows me the layers right
import torch
from torch.nn.modules.conv import Conv1d 

class ResidualBlock(nn.Module):
    # kernel_size = 3, stride = 1
    def __init__(self, nInput, nHidden, nOutput, dilations, kernel_size, causal, downsample = 0, no_skip_connections = False, no_bn2 = False):
        super(ResidualBlock,self).__init__()

        # variables
        # just need to initalize them, when I use them in the class, but not in the __init__ function!
        self.kernel_size = kernel_size
        self.causal = causal

        # layers
        self.relu = nn.ReLU()

        self.conv1 = self.build_conv_layer(nInput, nHidden, dilations[0])
        self.bn1 = nn.BatchNorm1d(nHidden)

        self.conv2 = self.build_conv_layer(nHidden, nOutput, dilations[1])
        if no_bn2:
            self.bn2 = None
        else:
            self.bn2 = nn.BatchNorm1d(nOutput)

        if no_skip_connections:
            self.skip_conv = None
        else:
            self.skip_conv = nn.Conv1d(nInput, nOutput, 1) #Skip Conv is added to output of ResidualBlock (later). Conv is used to reduce the size properly

        ### 
        # Downsampling in this paper AND at this place is never used [0 0 0 0], we just leave it here so that it is easier to incorporate a downsampling 
        # if we want to experiment with that later on
        if downsample > 0:
            self.downsample = nn.MaxPool1d(downsample, downsample)# TODO? kernel_size, stride?
        else:
            self.downsample = None 
        ### 

    def build_conv_layer(self, nInput, nOutput, dilation):
        if dilation >= 1:
            padding = dilation * (self.kernel_size - 1) # TODO? Berechnung?

            #causal (using only values from previous timesteps): Decoder
            if self.causal:
                pad_sides = (padding, 0)# just padding on the left side! ConstantPad1d((3, 1), 3.5) -> [ 3.5000,  3.5000,  3.5000,  1.6616,  1.4523, -1.1255,  3.5000]
            #non-causal: Encoder
            else:
                pad_sides = int(padding/2) #respectively on the left,right side half of the padding

            return nn.Sequential(
                nn.ConstantPad1d(pad_sides, 0), 
                nn.Conv1d(nInput, nOutput, self.kernel_size, dilation = dilation)
                )
        else:
            return nn.Conv1d(nInput, nOutput, self.kernel_size)

    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        #add residual connection values
        if self.skip_conv is not None: #TODO ohne is not None?
            out = out + self.skip_conv(x)
        
        out = self.relu(out)
        if self.bn2 is not None:
            out = self.bn2(out)
 
        # keep it in, as we perhaps add downsampling, GazeMAE didn't use it
        if self.downsample is not None:
            out = self.downsample(out)

        return out

#------------------------------------------------------------------------------------------------
class CausalBlock(ResidualBlock): #TODO eher DecoderBlock?
    def forward(self, x, representation=None):
        out = self.conv1(x)
        if representation is not None:
            out = out + representation.unsqueeze(-1) #Macro/Micro scale rep
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        if self.skip_conv is not None: 
            out = out + self.skip_conv(x) # Skip connection
        out = self.relu(out)
        if self.bn2 is not None:
            out = self.bn2(out)

        return out





        







