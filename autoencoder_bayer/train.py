
#tensorboard --logdir=runs
import time
import logging
from datetime import datetime

import numpy as np
from torch import manual_seed, nn, device, cuda, multiprocessing
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd

from data import get_corpora
from data.data import SignalDataset
from network.modelmanager import ModelManager
#from evaluate import *
#from evals.classification_tasks import *
#from evals.utils import *
from settings import *

from tqdm import tqdm

np.random.seed(RAND_SEED)
manual_seed(RAND_SEED)


class Trainer:
    def __init__(self, args):
        self.model = ModelManager(args)

        self.save_model = args.save_model
        
        self.device = device('cuda' if cuda.is_available() else 'cpu')
        self.model.network = self.model.network.to(self.device)

        # needed for Early Stopping
        self.best_loss = float('inf')
        self.counter = 0

        self.name_run = f"{args.hz}Hz_{args.signal_type}_ETRA_FIFA_MIT"
        # introduced two summary writer, so we can see the two functions in one graph in tensorboard
        self.tensorboard_train = SummaryWriter(f"runs/{self.name_run}_trainLoss")
        self.tensorboard_val = SummaryWriter(f"runs/{self.name_run}_valLoss")

        self._load_data(args)
        self._init_loss_fn(args)
        #self._init_evaluator()

    def _load_data(self, args):
        #TODO get_corpora: just add Datasets, which are higher than the given hz freq (args.hz)
        self.dataset = SignalDataset(get_corpora(args), args,
                                     caller='trainer')
        #print(dir(self.dataset.train_set))
        #print('Load data: ' , '\n' , self.dataset.train_set[0:30])

        _loader_params = {'batch_size': args.batch_size, 'shuffle': True,
                          'pin_memory': True,
                          'num_workers': 1,
                          'persistent_workers': True} #last two are new

        if len(self.dataset) % args.batch_size == 1:
            _loader_params.update({'drop_last': True})

        self.dataloader = DataLoader(self.dataset, **_loader_params) #train dataloader
        self.val_dataloader = (
            DataLoader(self.dataset.val_set, **_loader_params)
            if self.dataset.val_set else None)

    def _init_loss_fn(self, args):
        self._loss_types = ['total']

        # just to keep track of all the losses i have to log
        #TODO: do back? for what? self._loss_types.append('rec')
        self.loss_fn = nn.MSELoss(reduction='none')

    """
    def _init_evaluator(self):
        # for logging out this run
        _rep_name = '{}{}-hz:{}-s:{}'.format(
            run_identifier, 'mse',
            self.dataset.hz, self.dataset.signal_type)

        self.evaluator = RepresentationEvaluator(
            tasks=[Biometrics_EMVIC(), ETRAStimuli(),
                   AgeGroupBinary(), GenderBinary()],
            # classifiers='all',
            classifiers=['svm_linear'],
            args=args, model=self.model,
            # the evaluator should initialize its own dataset if the trainer
            # is using manipulated trials (sliced, transformed, etc.)
            dataset=(self.dataset if not args.slice_time_windows
                     else None),
            representation_name=_rep_name,
            # to evaluate on whole viewing time
            viewing_time=-1)

        if args.tensorboard:
            self.tensorboard = SummaryWriter(
                'tensorboard_runs/{}'.format(_rep_name))
        else:
            self.tensorboard = None
    """

    #saves loss every 100 batches
    def reset_running_loss_100(self):
        self.running_loss_100 = {'train': {l: 0.0 for l in self._loss_types}}#,
                             #'val': {l: 0.0 for l in self._loss_types}}
    
    def init_global_losses_100(self, num_checkpoints):
        self.global_losses_100 = {
            'train': {l: np.zeros(int(num_checkpoints * (int(len(self.dataloader)/TRAIN_SAVE_LOSS_EVERY_X_BATCHES)) ) +1) for l in self._loss_types},
            'val': {l: np.zeros(int(num_checkpoints * (int(len(self.dataloader)/TRAIN_SAVE_LOSS_EVERY_X_BATCHES)) ) +1) for l in self._loss_types}}
       
    #################################################################################
    #adds up to the epoch loss
    #self.running_loss[dset]['total']
    def init_running_loss(self):
        self.running_loss = {'train': {l: 0.0 for l in self._loss_types},
                             'val': {l: 0.0 for l in self._loss_types}}

    def init_global_losses(self, num_checkpoints):
        self.global_losses = {
            'train': {l: np.zeros(num_checkpoints) for l in self._loss_types},
            'val': {l: np.zeros(num_checkpoints) for l in self._loss_types}}
        
        #print(self.global_losses['train']['total'][0])

    #def update_global_losses(self, checkpoint, dset):
    #    self.global_losses[dset][self._loss_types[0]][checkpoint] = self.running_loss[dset]['total']

        #for dset in ['train', 'val']:
        #    for l in self._loss_types:# derzeit nur 'total', nicht mehr 'rec'
        #        self.global_losses[dset][l][checkpoint] = self.epoch_losses[dset][l]

        # for early stopping
    #    return self.global_losses[dset]['total'][checkpoint]

    def early_stopping(self, val_loss):
        stop = False
        if self.best_loss >= val_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter = self.counter + 1
        
        if self.counter > 5:
            print('Early stopping excecuted')
            stop = True
        
        return stop

    def batch_to_color(self, batch, name):
        ''' 
        #Without split 
        batch_old = pd.DataFrame(batch.reshape(20,-1).numpy())
        plt.matshow(batch_old)
        plt.colorbar()
        plt.show()
        '''
        fig, axs = plt.subplots(2)
        #shape (2,500) -> x/vel_x: (1,500) y/vel_y: (1,500)
        x = pd.DataFrame(batch[0,:].reshape(-1,50).numpy())
        y = pd.DataFrame(batch[1,:].reshape(-1,50).numpy())
        
        im1 = axs[0].matshow(x)
        fig.colorbar(im1, ax = axs[0], orientation = 'vertical')
        axs[0].set_title('x resp. vel_x', pad=30)
        im2 = axs[1].matshow(y)
        fig.colorbar(im2, ax = axs[1], orientation = 'vertical')
        axs[1].set_title('y resp. vel_y', pad=30)
        
        plt.savefig(f"{name}.png")
        

    def train(self, args, run_identifier):      
        print('################################# Start Training')
        self.init_global_losses(TRAIN_NUM_EPOCHS)
        self.init_global_losses_100(TRAIN_NUM_EPOCHS)
        counter_100 = 0
        
        
        t = tqdm(range(0, TRAIN_NUM_EPOCHS))
        #for e in tqdm(range(0, num_epochs), desc = 'Epochs'):
        for e in t:
            _checkpoint_start = time.time() 
            self.init_running_loss()
            self.reset_running_loss_100()
        
            #print(' \n train model')
            self.model.network.train()
            #for b, batch in enumerate(self.dataloader):
            for b, batch in enumerate(tqdm(self.dataloader, desc = 'Train Batches')):
                ###
                #if b == 0 and e == 0:
                #    self.tensorboard_train.add_graph(self.model.network, batch)
                ###
                self.model.optim.zero_grad()
                sample, sample_rec = self.forward(batch)
                #print(e*len(self.dataloader) + b)

                #save running loss 100 and reset it
                if (b+1) % TRAIN_SAVE_LOSS_EVERY_X_BATCHES == 0:
                    mean_loss = self.running_loss_100['train']['total'] / TRAIN_SAVE_LOSS_EVERY_X_BATCHES
                    self.global_losses_100['train']['total'][counter_100] = mean_loss                       #e*len(self.dataloader) + b
                    self.tensorboard_train.add_scalar(f"loss_per_{TRAIN_SAVE_LOSS_EVERY_X_BATCHES} batches", mean_loss, counter_100)
                    self.reset_running_loss_100()
                    counter_100 +=1

                    np.savetxt(f"original-batch-train", sample.numpy())
                    self.batch_to_color(sample, f"original-batch-train")
                    np.savetxt(f"reconstructed-batch-train", sample_rec.detach().numpy())
                    self.batch_to_color(sample_rec.detach(), f"reconstructed-batch-train")
                '''
                if b == 2:
                    break
                '''
            # save the train loss of the whole epoch
            self.log(e, 'train')

            #############################################################################
            # Validate the model
            #print('Validate Model')
            self.model.network.eval()
            for b, batch in enumerate(tqdm(self.val_dataloader, desc = 'Val Batches')):
                # In forward NN also calcs & saves the loss 
                sample_v, sample_rec_v = self.forward(batch) 
                print('val batch ', b)
                np.savetxt(f"original-batch-val", sample_v.numpy())
                self.batch_to_color(sample_v, f"original-batch-val")
                np.savetxt(f"reconstructed-batch-val", sample_rec_v.detach().numpy())
                self.batch_to_color(sample_rec_v.detach(), f"reconstructed-batch-val")
                
                #self.tensorboard.add_scalar('val loss', current_val_loss, e*len(self.val_dataloader) + b)
                '''
                if b == 2:
                    break
                '''
            self.log(e, 'val')

            t.set_postfix(loss = (self.global_losses['train']['total'][e], self.global_losses['val']['total'][e])) # print losses in tqdm bar
            # Save Model every epoch
            if self.save_model:
                #print(self.global_losses)
                #print(self.global_losses_100)
                self.model.save(e, run_identifier, self.global_losses, self.global_losses_100, self.name_run, args)
            _checkpoint_start = time.time()  

            # exit train loop, if early stopping says so
            '''
            stop = self.early_stopping(self.global_losses['val']['total'][e]) #current val loss!
            if stop:
                break
            '''
            

    def forward(self, batch):
        batch = batch.float()
        
        batch = batch.to(self.device)
        _is_training = self.model.network.training
        # out = [reconstructed batch, z]
        out = self.model.network(batch, is_training=_is_training)
        reconstructed_batch = out[0]         
        
        #loss = MSE(output, target)
        loss = self.loss_fn(reconstructed_batch, batch
                            ).reshape(reconstructed_batch.shape[0], -1).sum(-1).mean()
        dset = 'train' if self.model.network.training else 'val'
        self.running_loss[dset]['total'] += loss.item()

        if dset == 'val':
            print('loss val', loss)    

        #update network if we are training
        if self.model.network.training:
            self.running_loss_100[dset]['total'] += loss.item()
            loss.backward()
            self.model.optim.step()

        rand_idx = np.random.randint(0, batch.shape[0])
        return batch[rand_idx].cpu(), reconstructed_batch[rand_idx].cpu()

    '''
    def evaluate_representation(self, sample, sample_rec, i):
        if sample is not None:
            viz = visualize_reconstruction(
                sample, sample_rec,
                filename='{}-{}'.format(run_identifier, i),
                loss_func=self.rec_loss,
                title='[{}] [i={}] vl={:.2f} vrl={:.2f}'.format(
                    self.rec_loss, i, self.epoch_losses['val']['total'],
                    self.epoch_losses['val']['rec']),
                savefig=False if self.tensorboard else True)

            if self.tensorboard:
                self.tensorboard.add_figure('e_{}'.format(i),
                                            figure=viz,
                                            global_step=i)

        self.evaluator.extract_representations(i, log_stats=True)
        scores = self.evaluator.evaluate(i)
        if self.tensorboard:
            for task, classifiers in scores.items():
                for classifier, acc in classifiers.items():
                    self.tensorboard.add_scalar(
                        '{}_{}_acc'.format(task, classifier), acc, i)
    
    '''
    def log(self, e, dset):
        def get_mean_losses():
            try:
                iters = (len(self.dataloader) if dset == 'train'     
                         else len(self.val_dataloader))
            except TypeError:
                iters = 1
            return (self.running_loss[dset]['total'] / iters)

        def to_tensorboard(dset, loss):
            self.tensorboard.add_scalar(f'{dset}_loss_per_epoch', loss, e)
        
        if dset == 'train':
            tr_loss = get_mean_losses()
            # save the mean loss of this epoch during training
            self.global_losses[dset]['total'][e] = tr_loss
            # reset running loss for the epoch
            self.running_loss[dset]['total'] = 0.0
            #string = '[amount batches {}] train_loss: {:.4f}, ({:.2f}s)'.format((e+1)*len(self.dataloader), tr_loss) # [0/433] TLoss: 2.6610
            if self.tensorboard_train:
                #to_tensorboard('train', tr_loss)
                self.tensorboard_train.add_scalar(f'loss_per_epoch', tr_loss, e)
        elif dset == 'val':
            print('In log ')
            print('Running loss ', self.running_loss)
            val_losses = get_mean_losses()
            self.global_losses[dset]['total'][e] = val_losses
            self.running_loss[dset]['total'] = 0.0
            #string = '[amount batches {}] VLoss: {:.4f} ({:.2f}s)'.format((e+1)*len(self.val_dataloader), val_losses)
            if self.tensorboard_val:
                print('Mean loss ', val_losses)
                self.tensorboard_val.add_scalar(tag = f'loss_per_epoch', scalar_value = val_losses, global_step = e)
        #logging.info(string)


def main():
    args = get_parser().parse_args()
    run_identifier = args.signal_type                   #TODO datetime.now().strftime('%m%d-%H%M')
    """
    setup_logging(args, run_identifier)
    print_settings()

    logging.info('\nRUN: ' + run_identifier + '\n')
    logging.info(str(args))
    """
    
    multiprocessing.freeze_support()

    trainer = Trainer(args)
    trainer.train(args, run_identifier)

if __name__ == "__main__":
    main()
