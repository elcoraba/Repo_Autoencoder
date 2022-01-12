## just copied
import logging
import time
import pickle
from os import path

import h5py
import numpy as np
from numpy.core.numeric import NaN
import pandas as pd

import data.utilsData as du
from settings import *

import pandas as pd

# hier laden und preprocessing
# class for loading the datasets
class EyeTrackingCorpus:
    def __init__(self, args=None):
        if args:
            self.signal_type = args.signal_type
            self.effective_hz = args.hz or self.hz              #The hz I give in the terminal, e.g. FIFA: self.hz = 1000, python train.py ... -hz = 30 -> effective hz

            self.slice_time_windows = args.slice_time_windows
            if self.slice_time_windows:
                assert self.slice_time_windows in [None, False,
                                                   '2s-overlap', '2s-disjoint']
                self.slice_length = self.effective_hz * int(self.slice_time_windows[0])
                self.viewing_time = 0  # process whole signal
            else:
                self.viewing_time = args.viewing_time

        self.root = DATA_ROOT + self.root
        self.stim_dir = DATA_ROOT + self.stim_dir if self.stim_dir else None

        self.name = self.__class__.__name__
        self.dir = GENERATED_DATA_ROOT + self.name
        self.data = None

    def load_data(self, load_labels=False):
        print('In load data')
        logging.info('\n---- Loading {} data set with {}-sliced {} signals...'.format(
            self.name, self.slice_time_windows, self.signal_type))
        if self.slice_time_windows:
            # load hdf5 file because it's too much to load into memory
            self.hdf5_fname = self.dir + '-slices-{}-{}-{}hz'.format(
                self.signal_type, self.slice_time_windows, self.effective_hz)

            logging.info('Using hdf5 for time slices. ' +
                         'Filename: {}'.format(self.hdf5_fname + '.hdf5'))

            if path.exists(self.hdf5_fname + '.hdf5'):
                hdf5_dataset = h5py.File(self.hdf5_fname + '.hdf5', 'r')['slices']
                logging.info('Found {} slices'.format(len(hdf5_dataset)))
                self.data = hdf5_dataset
                return
            else:
                logging.info('hdf5 file not found.')

        self.load_raw_data()
        
    def load_raw_data(self):
        print('In load raw data')
        logging.info('Extracting raw data...'.format(self.name))

        data_file = self.dir + '-data.pickle'
        if not path.exists(data_file):
            extract_start = time.time()
            self.data = pd.DataFrame(
                columns=['subj', 'stim', 'task', 'timestep', 'x', 'y'],
                data=self.extract())
            self.data.x = self.data.x.apply(lambda a: np.array(a))
            self.data.y = self.data.y.apply(lambda a: np.array(a))
            logging.info('- Done. Found {} samples. ({:.2f}s)'.format(
                len(self.data),
                time.time() - extract_start))
            with open(data_file, 'wb') as f:
                pickle.dump(self.data, f)
        else:
            with open(data_file, 'rb') as f:
                self.data = pickle.load(f)
            logging.info('- Data loaded from' + data_file)

        self.preprocess_data()

    def extract(self):
        """
        Should be implemented by all data sets.
        Go through all samples and return a NumPy array of size (N, 4)
        N: number of samples (subjects x stimuli)
        4: columns for the pd DataFrame (subj, stim, x coords, y coords)
        """
        print('In extract')
        pass

    def append_to_df(self, data):
        self.data = self.data.append(
            dict(zip(['subj', 'stim', 'x', 'y'], list(data))),
            ignore_index=True)

    def __len__(self):
        return len(self.data)

    def preprocess_data(self):
        def preprocess(trial):
            #print('In preprocess')
            # trim to specified viewing time
            # no trim happens when self.slice_time_windows
            trial.x = trial.x[:sample_limit]
            trial.y = trial.y[:sample_limit]
            
            #--------
            # always convert blinks (negative values) to Nan
            trial.x[np.where(trial.x < 0)] = float("nan") # was = 0
            trial.y[np.where(trial.y < 0)] = float("nan")

            # coordinate normalization is not necessary for velocity space
            if self.signal_type == 'pos':
                # trial = self.pull_coords_to_zero(trial)
                trial.x = np.clip(trial.x, a_min=0, a_max=self.w or MAX_X_RESOLUTION)
                trial.y = np.clip(trial.y, a_min=0, a_max=self.h or MAX_Y_RESOLUTION)

            trial.x = du.interpolate_nans(trial.x)
            trial.y = du.interpolate_nans(trial.y)

            # scale coordinates so 1 degree of visual angle = 35 pixels
            try:
                scale_value = PX_PER_DVA / self.px_per_dva
                trial[['x', 'y']] *= scale_value
            except AttributeError:  # if corpora has no information about dva
                pass

            # Clip x and y values between 0 and 1
            trial.x = trial.x/self.w                                        #[492.82, 488.98, -> [0.48126953125, 0.47751953125,
            trial.y = trial.y/self.h
            # As we just do the clipping when we have 'pos' here would faults appear when when don't include the if
            if self.signal_type == 'pos':
                assert np.all(trial.x >= 0) and np.all(trial.x <= 1), 'Problem! A x-Value in the dataset is not between 0 and 1'   + 'Trial: ' + trial.subj + ' Stimulus: ' + trial.stim + ' Value below 0: ' + str(np.argwhere(trial.x < 0)) + ' Value above 1: ' + str(np.argwhere(trial.x > 1))
                assert np.all(trial.y >= 0) and np.all(trial.y <= 1), 'Problem! A y-Value in the dataset is not between 0 and 1: ' + 'Trial: ' + trial.subj + ' Stimulus: ' + trial.stim + ' Value below 0: ' + str(np.argwhere(trial.y < 0)) + ' Value above 1: ' + str(np.argwhere(trial.y > 1))       

            #Sybille: preprocessing
            trial = self.validate_data(trial) 

            if self.resample == 'down':
                trial = du.downsample(trial, self.effective_hz, self.hz)
            elif self.resample == 'up':
                trial = du.upsample(trial, self.effective_hz, self.hz)

            #Do clipping again, just to be sure up or downsampling didn't do something crazy
            if self.signal_type == 'pos':
                assert np.all(trial.x >= 0) and np.all(trial.x <= 1), 'Problem! A x-Value in the dataset is not between 0 and 1'   + 'Trial: ' + trial.subj + ' Stimulus: ' + trial.stim + ' Value below 0: ' + str(np.argwhere(trial.x < 0)) + ' Value above 1: ' + str(np.argwhere(trial.x > 1))
                assert np.all(trial.y >= 0) and np.all(trial.y <= 1), 'Problem! A y-Value in the dataset is not between 0 and 1: ' + 'Trial: ' + trial.subj + ' Stimulus: ' + trial.stim + ' Value below 0: ' + str(np.argwhere(trial.y < 0)) + ' Value above 1: ' + str(np.argwhere(trial.y > 1))       


            return trial

        sample_limit = (int(self.hz * self.viewing_time)
                        if self.viewing_time > 0
                        else None)

        if (self.hz - self.effective_hz) > 10:
            self.resample = 'down'
            logging.info('Will downsample {} to {}.'.format(
                self.hz, self.effective_hz))
        elif (self.effective_hz - self.hz) > 10:
            self.resample = 'up'
            logging.info('Will upsample {} to {}.'.format(
                self.hz, self.effective_hz))
        else:
            self.resample = None

        self.data = self.data.apply(preprocess, 1)
        print('Data/trial columns ', self.data.columns)

        if 'vel' in self.signal_type:
            #('OLD v: ', self.data['v']/ (1000 / self.effective_hz))
            logging.info('Calculating velocities...')

            #(x_old - x_new/diff timestep) # with variable timesteps 
            self.data['v'] = self.data[['x', 'y']].apply(
                lambda x: np.abs(np.diff(np.stack(x))).T, 1)
            div = self.data[['timestep']].apply(lambda x: np.abs(np.diff(np.stack(x))).T, 1)
            self.data['v'] = self.data['v'].divide(div) # TODO stack und abs entfernen?
            

            ####FIFA
            #v: [[0.0, 0.000996946495305795], [0.0005607824036... (3200,0), v[0].shape = (2020,2) (cols, rows)
            #div: [[1], [1], [1], [1],... (3200,0), div[0].shape = (2020,1)
            #final v: [[0.0, 0.000996946495305795], [0.0005607824036... (3200,0)  
            #####

        ###  Throw out NaN values
        x_nan_idx = self.data['x'].isna()                   #self.data is a Dataframe, self.data['x'] is a Series
        y_nan_idx = self.data['y'].isna()
        #if sum(y_nan_idx)> 0:
        #    print('Nans ')
        idx = (x_nan_idx | y_nan_idx)
        self.data['x'] = self.data['x'][~idx]
        self.data['y'] = self.data['y'][~idx]
        #####################            

        if self.slice_time_windows:
            hdf5_dataset = self.write_slices_to_h5()
            self.data = hdf5_dataset

    def slice_trials(self):
        def _slice(trial, trial_num):
            copies = []

            for slice_start in range(0, len(trial.x), increment):
                copy = trial.copy()
                copy['trial_num'] = trial.name  # trial number
                if self.signal_type == 'vel':
                    copy.x = trial.v[slice_start: slice_start + self.slice_length, 0]
                    copy.y = trial.v[slice_start: slice_start + self.slice_length, 1]
                else:
                    copy.x = trial.x[slice_start: slice_start + self.slice_length]
                    copy.y = trial.y[slice_start: slice_start + self.slice_length]

                # the slice should at least be half as long as the slice length
                if slice_start == 0 or len(copy.x) >= int(self.slice_length / 2):
                    copies.append(copy)

            return copies

        logging.info('Slicing samples into time windows...')

        if self.slice_time_windows == '2s-overlap':
            increment = int(self.slice_length * SLICE_OVERLAP_RATIO)
        elif self.slice_time_windows == '2s-disjoint':
            increment = self.slice_length

        # dump unprocessed samples here
        return pd.DataFrame([k for i, trial in self.data.iterrows()
                             for k in _slice(trial, i)])

    def write_slices_to_h5(self):
        def iter_slice_chunks(df, chunk_size=50000):
            chunk_start = 0
            while chunk_start < len(df):
                chunk = df.iloc[chunk_start: chunk_start + chunk_size]
                yield np.stack(chunk.apply(
                    lambda x: du.pad(self.slice_length,
                                     np.stack(x[['x', 'y']]).T), 1))
                chunk_start += chunk_size

        slices_df = self.slice_trials()

        # prepare hdf5 file. maxshape is resizable.
        data_dim = self.slice_length
        hdf5_file = h5py.File(self.hdf5_fname + '.hdf5', 'w')

        hdf5_dataset = hdf5_file.create_dataset(
            'slices', (1, data_dim, 2),
            maxshape=(None, data_dim, 2))

        # process the raw slices then write to file by chunks
        for slice_chunk in iter_slice_chunks(slices_df):
            curr_size = len(hdf5_dataset)
            curr_size = curr_size if curr_size != 1 else 0
            hdf5_dataset.resize(curr_size + len(slice_chunk), axis=0)
            hdf5_dataset[curr_size:] = slice_chunk

        # Store each time window's corresponding info using pandas
        # can be used for large-scale classification later on.
        slices_df.drop(['x', 'y'], axis=1, inplace=True)
        slices_df.to_hdf(self.hdf5_fname + '.h5', key='df')

        logging.info('Saved {} slices to {}'.format(
            len(slices_df), self.hdf5_fname))
        return hdf5_dataset

######################################################################NEW###
    def validate_data(self, trial):
        #print('In validate data')
        #self.sampling_rate()
        #Adapt trial, if sampling freq is not continous - add timestamps in between
        trial = self.change_in_sampling_freq(trial)

        return trial
    
    #The calculated sampling frequency in subject  JP and trial  0103 differs from the given frequency 1000
    #Calculated sampling frequency:  951.9564140663695
    #Change in sampling freq.:  subj:  JP trial:  0098 timestamp:  18703378   18703385
    #Matlab: exp 2.000000 trial 74.000000 t 431.000000              -> trial 0103 ist an 74ster Stelle von self.stimuli

    #Fill holes in dataset before any up/downsampling is done
    def change_in_sampling_freq(self, trial):
        point_pairs = []
        hole_array = []
        for i in range(len(trial.timestep) - 1):
            if trial.timestep[i] + self.step == trial.timestep[i+1]:
                continue
            else:
                point_pairs.append((trial.timestep[i], trial.timestep[i+1]))
        # Were there any holes in the data found? Then do the upsampling
        if len(point_pairs) > 0: 
            
            #if trial['subj'] == 'DA' and trial['stim'] == '0219.jpg':
            #    np.savetxt(f"FIFA_holes_subj {trial['subj']}_stim {trial['stim']}_x.csv", list(zip(trial['timestep'], trial['x'])), delimiter=',')
            #    np.savetxt(f"FIFA_holes_subj {trial['subj']}_stim {trial['stim']}_y.csv", list(zip(trial['timestep'], trial['y'])), delimiter=',')
            #    exit()

            print('Adapt sampl.freq. ', 'subj: ', trial.subj, 'stim: ', trial.stim)
            trial = du.upsample_between_timestamp_pairs(trial, self.effective_hz, self.hz, point_pairs, self.step) 

            ##### find holes
            #temp = [trial.subj, trial.stim, point_pairs]
            #hole_array.append(temp) 
            #print('Hole Array ', hole_array)
            # 'WS', '0085.jpg', 'WS', '0065.jpg', 'WS', '0120.jpg', 'WS', '0117.jpg', 'WS', '0077', '0066.jpg'
            # 'TS', '0122.jpg', '0150.jpg', '0050.jpg'
            # 'DA', '0034.jpg', '0126.jpg', '0164.jpg', '0200.jpg', '0219.jpg'
        
        return trial




