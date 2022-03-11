import argparse
from fileinput import filename
import numpy as np
from sklearn.preprocessing import StandardScaler
#from torch import no_grad, Tensor, manual_seed
import torch
from torch import manual_seed, nn, device, cuda, multiprocessing
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.data import SignalDataset

from datetime import datetime
from network.autoencoder import Autoencoder
from settings import *
import json

from data.datasets import *
from eval_classifier_settings import *
from eval_classification_tasks import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, GridSearchCV

from tqdm import tqdm

np.random.seed(RAND_SEED)
manual_seed(RAND_SEED)


class AccuracyEvaluator:
    def __init__(self, tasks): #args , tasks, classifiers='all', args=None, **kwargs):
        self.device = device('cuda' if cuda.is_available() else 'cpu')
        #TODO 
        #self.network = self.network.to(self.device)
        self._init_loss_fn()
        self.tasks = tasks

        self.classifiers = list(CLASSIFIER_PARAMS.values())
        self.name_run = 'test_eval'
        self.tensorboard = SummaryWriter(f"runs_eval/{self.name_run}_evalLoss") #TODO which loss do we use?

        #TODO 
        self.new_df = pd.DataFrame([], columns=['data', 'corpus', 'subj', 'stim', 'z']) # ['corpus', 'subj', 'stim', 'task']
        

    def loadModel(self, model_name):
        #######
        #load model
        model = torch.load('../models/' + model_name)# lädt aufbau des netzwerkes #pos-i3738 = model_name
        try:
            network = model['network']
        except KeyError:
            network = model['model']

        network.load_state_dict(model['model_state_dict']) # lädt gewichte von netzwerk
        ########
        #load params
        with open('../models/' + model_name + '.json', 'r') as jasonfile:
                params_savedModel = jasonfile.read()
        # convert string back to a dict
        params_savedModel = json.loads(params_savedModel)
        #print(params_savedModel['pos or vel'])
        
        self.network = {} # leeres dict
        if params_savedModel['signal_type'] == 'vel':
            self.network['vel'] = network.eval()
        elif params_savedModel['signal_type'] == 'pos':
            self.network['pos'] = network.eval()
        return self.network, params_savedModel

    def loadData(self, list_datasets, args):
        # Now we don't divide in train and val set
        self.dataset = SignalDataset(list_datasets, args, caller='')
        self.new_data = self.dataset.new_data
       
        _loader_params = {'batch_size': args['bs'], 'shuffle': True,
                          'pin_memory': True,
                          'num_workers': 1,
                          'persistent_workers': True} #last two are new

        if len(self.dataset) % args['bs'] == 1:
            _loader_params.update({'drop_last': True})

        self.dataloader = DataLoader(self.dataset, **_loader_params) #train dataloader, bzw for evaluation just this loader, as val_loader is empty
        self.val_dataloader = (
            DataLoader(self.dataset.val_set, **_loader_params)
            if self.dataset.val_set else None)

        # To get the values of df for evaluation
        # TODO how does it look like, concatinieren
        self.new_data = self.dataset.new_data

    
    def _init_loss_fn(self):
        self._loss_types = ['total']

        # just to keep track of all the losses i have to log
        #TODO: do back? for what? self._loss_types.append('rec')
        self.loss_fn = nn.MSELoss(reduction='none')

    def loop(self, dset):
        # data loader
        for b, batch in enumerate(tqdm(self.dataloader, desc = 'All Batches')):
                # In forward NN also calcs & saves the loss 
                sample, sample_rec = self.forward(batch, dset) 
                #np.savetxt(f"original-batch-val", sample.numpy())
                #np.savetxt(f"reconstructed-batch-val", sample_rec.detach().numpy())
                
                #self.tensorboard.add_scalar('val loss', current_val_loss, e*len(self.val_dataloader) + b)
                '''
                if b == 2:
                    break
                '''
            #self.log(e, 'val')

    
    def forward(self, batch, dset):
        batch = batch.float()
        
        batch = batch.to(self.device)
        _is_training = False #self.model.network.training
        # out = [reconstructed batch, z]
        out = self.network[str(dset)](batch, is_training=_is_training) #TODO was self.model.network
        reconstructed_batch = out[0]    
        
        #loss = MSE(output, target)
        loss = self.loss_fn(reconstructed_batch, batch
                            ).reshape(reconstructed_batch.shape[0], -1).sum(-1).mean()
        #dset = 'train' if self.model.network.training else 'val'   
        # TODO brauche ich das?     
        #self.running_loss[dset]['total'] += loss.item()  
        #self.running_loss_100[dset]['total'] += loss.item()   

        # #####################
        z = out[1]
        self.new_df['z'] = z
        # #####################  

        rand_idx = np.random.randint(0, batch.shape[0])
        return batch[rand_idx].cpu(), reconstructed_batch[rand_idx].cpu()

    def evaluate(self):
        scores = {}
        for i, task in enumerate(self.tasks):
            scores[task] = {}
            z, l = task.get_zl()
            #self.classifier: ('svm_linear', SVC() ),{'svm_linear__kernel': ['linear'], 'svm_linear__gamma': ['auto'], ....}
            pipeline = Pipeline([('scaler', StandardScaler()), self.classifiers])
            grid_crossval = GridSearchCV(pipeline, self.classifiers[1], cv = 5, n_jobs = 4, scoring = ['accurarcy'], refit = 'accurarcy' )
            grid_crossval.fit(self.new_df, l) #TODO z mit was rein? stack?
            # acc = grid_cv.cv_results_['mean_test_accuracy'].max()
            accurarcy = grid_crossval.cv_results_
            print('acc ', accurarcy)

            scores[task] = accurarcy
        
        return scores
 
################################################################################
        


CORPUS_LIST = {
    #'Cerf2007-FIFA': Cerf2007_FIFA,
    #'ETRA2019': ETRA2019,
    'MIT-LowRes': MIT_LowRes,
    #'GAZEBASE': GAZEBASE
}
#'EMVIC2014': EMVIC2014,

frequencies = {
    #1000: ['GAZEBASE'],
    250: ['MIT-LowRes'],
    #1000: ['Cerf2007-FIFA']#, 'GAZEBASE'],
    #500: ['ETRA2019'],
    # the ones from MIT are 240Hz but oh well  
}
#1000: ['EMVIC2014', 'Cerf2007-FIFA'], 

def get_corpora_evaluation(args, additional_corpus=None):
    corpora = []
    for f, c in frequencies.items():
        corpora.extend(c)
    
    return {c: CORPUS_LIST[c](args) for c in corpora}


if __name__ == '__main__':
    run_identifier = 'eval_' + datetime.now().strftime('%m%d-%H%M')
    #args = get_parser().parse_args()                                #(augment=False, batch_size=64, epochs=200, hz=0, learning_rate=0.0005, model_pos='', model_vel='', save_model=True, signal_type='pos', slice_time_windows=None, tensorboard=True, use_validation_set=True, viewing_time=-1)
    tasks = [AgeGroupBinary(), GenderBinary()]
    evaluator = AccuracyEvaluator(tasks)
    model_filename = r'vel-e1-hz250'
    network, params = evaluator.loadModel(model_filename)           #TODO: add all arguments to saving of model and change val or pos to signal_type
    # In corpusdata und data die init fkt angepasst, da hier params dict sind und nicht dieser komische args Type
    args = params
    # Which datasets should be loaded # TODO kein up/downsampling?
    list_datasets = get_corpora_evaluation(args)
    evaluator.loadData(list_datasets, args)
    dset = args['signal_type']
    evaluator.loop(dset)
    #evaluator.extract_representations()
    evaluator.evaluate()