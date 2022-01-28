# __init__

import logging
import json
from datetime import datetime, date
import numpy as np
from settings import *
import csv

import torch

from .autoencoder import Autoencoder

# wird in train aufgerufen
class ModelManager:
    def __init__(self, args, training=True):
        self.is_training = training
        self.load_network(args)

    def load_network(self, args):
        # if we want to load network
        if args.model_pos or args.model_vel:# is string empty or not, name des models e.g. #pos-i3738
            if self.is_training:
                self.network, self.optim = self._load_pretrained_model(args.model_pos or args.model_vel)
            else: #evaluation
                self.network = {} # leeres dict
                vel_net = self._load_pretrained_model(args.model_vel)
                if vel_net: # None, wenn model_name nicht vorhanden in Model Ordner
                    self.network['vel'] = vel_net.eval() # packe key in dict, was netzwerk enthält, netzwerk in eval modus versetzen (sind nicht im traiuning)
                pos_net = self._load_pretrained_model(args.model_pos)
                if pos_net:
                    self.network['pos'] = pos_net.eval()
        # erstelle network
        else:
            self.network = Autoencoder(args)

            self.optim = torch.optim.Adam(self.network.parameters(),
                                          lr=args.learning_rate)
            # Shows structure of Network - add it later again
            #self._log(self.network)
            print('log network')

    def _load_pretrained_model(self, model_name):
        if not model_name:
            return None

        logging.info('Loading saved model {}...'.format(model_name)) 
        model = torch.load('../models/' + model_name)# lädt aufbau des netzwerkes #pos-i3738 = model_name
        try:
            network = model['network']
        except KeyError:
            network = model['model']
        self._log(network)

        network.load_state_dict(model['model_state_dict']) # lädt gewichte von netzwerk

        if self.is_training:
            optim = torch.optim.Adam(network.parameters())
            optim.load_state_dict(model['optimizer_state_dict'])
            return network, optim

        return network

    def _log(self, network):
        print('\n ' + str(network)) # logging.info
        print('# of Parameters: ' +
                     str(sum(p.numel() for p in network.parameters()
                             if p.requires_grad)))

    # wird in train aufgerufen
    def save(self, e, run_identifier, losses, losses_100, name_run, args): 
        model_filename = '../models/' + run_identifier + '-e' + str(e) + '-hz' + str(args.hz) #pos-i3738, i = iteration
        torch.save(
            {
                'epoch': e,
                'network': self.network,
                'model_state_dict': self.network.state_dict(), #a dictionary containing a whole state of the module
                'optimizer_state_dict': self.optim.state_dict(),
                'losses': losses
            }, model_filename)
        logging.info('Model saved to {}'.format(model_filename))

        #save params in extra file
        params = {
            "iteration"     : run_identifier,
            "pos or vel"    : args.signal_type,               
            "lr"            : args.learning_rate,
            "hz"            : args.hz,
            "viewing time"  : args.viewing_time,
            "bs"            : args.batch_size,      #I added
            "last loss"     : list(losses)[-1],     #before: losses[-1],
            "current day"   : date.today().strftime("%d.%m.%Y"),
            "current time"  : datetime.now().strftime("%H:%M:%S"),
            "epoch losses"  : list(losses),
            "epoch losses 100": list(losses_100)          # losses every 100 batches
        }

        jsonFile = json.dumps(params) #was .dump()

        with open(model_filename, 'w') as jasonfile: # function opens a file, and returns it as a file object., Write - Opens a file for writing, creates the file if it does not exist
            jasonfile.write(jsonFile)

        
