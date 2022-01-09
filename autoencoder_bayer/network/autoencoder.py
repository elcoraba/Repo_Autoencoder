import logging

from torch import nn, manual_seed, cat, exp, randn_like, randn
import torch 

from network.encoder import Encoder
from network.decoder import Decoder

from settings import *

manual_seed(RAND_SEED)

class Autoencoder(nn.Module):
    def __init__(self, args):                               
        super(Autoencoder, self).__init__()
        self.latent_size = LATENT_SIZE
        #self.latent_size = int(LATENT_SIZE / 2) # TODO Thomas

        #kernel_size, filters, dilations, downsamples, causal, nInput = 2
        self.encoder = Encoder(ENCODER_KERNEL_SIZE, ENCODER_FILTERS, ENCODER_DILATIONS, ENCODER_DOWNSAMPLE, False)
        #hz, viewing_time, kernel_size, filters, dilations, input_dropout, latent_dim  
        self.decoder = Decoder(args, DECODER_KERNEL_SIZE, DECODER_FILTERS, DECODER_DILATIONS, DECODER_INPUT_DROPOUT, self.latent_size) #TODO HZ and vt

        self.bottleneck_fns = nn.ModuleDict(
            {
                '1': self.get_bottleneck(in_dim = self.encoder.out_dim1), 
                '3': self.get_bottleneck(in_dim = self.encoder.out_dim3) 
            }
        )

    # erstes FC
    def get_bottleneck(self, in_dim):
        return nn.Sequential(
            nn.Linear(in_dim, self.latent_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.latent_size))
    

    def bottleneck(self, x, level = '1'):
        # for when batch size = 1
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = x.reshape(x.shape[0], -1)

        fc_fn = self.bottleneck_fns[level]
        #prev_z weg
        z = fc_fn(x)

        return z                                               
    

    def encode(self, x):                                     
        micro, macro = self.encoder(x)
        z1 = self.bottleneck(micro, level = '1')        # erstes FC # [128, 64]
        z2 = self.bottleneck(macro, level = '3')

        z1 = torch.unsqueeze(z1, 0) # [1, 128, 64]
        z2 = torch.unsqueeze(z2, 0)
 
        z = cat([z2, z1], 0) #tensor
        
        return z
    

    def forward(self, x, is_training=False):
        z = self.encode(x)                      
        out = self.decoder(z, x, is_training)   #forward: z, x_true, is_training

        return out, z
    





