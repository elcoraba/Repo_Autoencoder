from numpy import random
from torch import nn, cat, zeros

from network.residualblock import CausalBlock

from settings import *
random.seed(RAND_SEED)

class Decoder(nn.Module):
    def __init__(self, args, kernel_size, filters, dilations, input_dropout, latent_dim):  
        super(Decoder, self).__init__()

        self.nInput = 2 
        self.causal = True

        #self.latent_projections = nn.ModuleList([])     
        self.input_dropout = input_dropout                          # prob for dropout, e.g. p = 0.5

        self.in_dim = int(args.hz *args.viewing_time)               # Anzahl der Datenpunkt #Vir: 250 * 2
        self.filters = filters
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.dec = nn.ModuleList([])                                # list of layers

        self.dec = nn.ModuleList(
            # nInput, nHidden, nOutput, dilations, kernel_size, causal, downsample = 0, no_skip_connections = False
            [CausalBlock(self.nInput, self.filters[0], self.filters[0], self.dilations[0], self.kernel_size, self.causal, no_bn2 = False), 
            CausalBlock(self.filters[0], self.filters[1], self.filters[1], self.dilations[1], self.kernel_size, self.causal, no_bn2 = False), 
            CausalBlock(self.filters[1], self.filters[2], self.filters[2], self.dilations[2], self.kernel_size, self.causal, no_bn2 = False), 
            CausalBlock(self.filters[2], self.filters[3], self.nInput, self.dilations[3], self.kernel_size, self.causal, no_bn2 = True)] 
        )

        self.microRep = nn.Linear(latent_dim,self.filters[0])       # second FC
        self.macroRep = nn.Linear(latent_dim,self.filters[2])

    # z: [z_1, z_2], x_true: (true) input - what encoder gets as input
    def forward(self, z, x_true, is_training):
        # pad the input at its left so there is no leak from input t=1 to
        # output t=1. should be: output for t=1 is dependent on input t=0
        x = cat((x_true, zeros(x_true.shape[0], 2, 1)), dim=2) # .cuda()
        # Destroy input - but just during training!
        x = nn.functional.dropout(x, self.input_dropout, is_training)

        for (i,block) in enumerate(self.dec):
            latent_proj = None
            if i == 0:
                latent_proj = self.macroRep(z[0])
            elif i == 2:
                latent_proj = self.microRep(z[1])
            
            if latent_proj is not None:
                x = block(x, latent_proj)                             # add representation to convolution
            else:
                x = block(x)
        

        return x[:,:,:-1]
            
                  




        
