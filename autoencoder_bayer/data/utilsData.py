## just copied 
from scipy import io
from scipy.interpolate import interp1d
import os
import pandas as pd
import numpy as np


def load(filename, file_format, **kwargs):
    if file_format == 'matlab':
        return io.loadmat(filename, squeeze_me=True)
    if file_format == 'excel':
        return pd.read_excel(filename, sheet_name=kwargs['sheet'])
    if file_format == 'csv':
        return pd.read_csv(filename,
                           **kwargs)


def listdir(directory):
    # a wrapper just to prepend DATA_ROOT
    return [f for f in os.listdir(directory)
            if not f.startswith('.')]


def pad(num_gaze_points, sample):
    sample = np.array(sample)
    num_zeros = num_gaze_points - len(sample)
    return np.pad(sample,
                  ((0, num_zeros), (0, 0)),
                  constant_values=0)


def interpolate_nans(trial):
    nans = np.isnan(trial)
    if not nans.any():
        return trial
    nan_idxs = np.where(nans)[0]
    not_nan_idxs = np.where(~nans)[0]
    not_nan_vals = trial[not_nan_idxs]
    trial[nans] = np.interp(nan_idxs, not_nan_idxs, not_nan_vals)
    return trial


def pull_coords_to_zero(coords):
    non_neg = coords.x >= 0
    coords.x[non_neg] -= coords.x[non_neg].min()
    non_neg = coords.y >= 0
    coords.y[non_neg] -= coords.y[non_neg].min()
    return coords

def downsample_old(trial, new_hz, old_hz):
    skip = int(old_hz / new_hz)
    trial.x = trial.x[::skip]
    trial.y = trial.y[::skip]
    return trial

#-------------------------------------------
#Downsample, old Hz:  1000
#Downsample, new Hz:  749.8761763249133
#1000 -> 750Hz: 0.  1.  3.  4.  5.  7.  8.  9. 11. 12. 13. 15. 16. 17. 19.
#new Hz: 299.95042141794744
#1000 -> 300Hz: 0.  3.  7. 10. 13. 17. 20. 23. 27. 30. 33. 37. 40. 43. 47.
def downsample(trial, new_hz, old_hz):
    #########For the comparison, type(trial['x'] -> np array
    #print('In downsample ', trial['stim'])
    #if trial['subj'] == '022' and trial['stim'] == 'WALDO/wal004.bmp':
    #    np.savetxt(f"ETRA_500Hz_subj {trial['subj']}_stim {trial['stim'][-10:]}_x.csv", list(zip(trial['timestep'], trial['x'])), delimiter=',')
    #    np.savetxt(f"ETRA_500Hz_subj {trial['subj']}_stim {trial['stim'][-10:]}_y.csv", list(zip(trial['timestep'], trial['y'])), delimiter=',')
    #########
    step = 1000/new_hz                                      # hier darf man nicht runden! Sonst machen wir wieder nur 1, 2, ... Schritte
    #FIFA: max_timestep = trial['timestep'][-1]
    #if type(trial['timestep']) #TODO continue
    max_timestep = trial['timestep'].iloc[-1]
    new_timesteps = np.arange(0, max_timestep, step)
    new_timesteps = new_timesteps.round().astype(int)
 
    idx = np.intersect1d(trial.timestep, new_timesteps, return_indices=True)[1]                                               
    trial.timestep = trial.timestep[idx]
    trial.x = trial.x[idx]
    trial.y = trial.y[idx]

    #########For the comparison, type(trial['x'] -> np array
    #NATURAL/nat014
    #022, PUZZLE/puz010.bmp
    #if trial['subj'] == '022' and trial['stim'] == 'WALDO/wal004.bmp':
    #    np.savetxt(f"ETRA_500Hz_to_30Hz_Downsample_wo_interpol_subj {trial['subj']}_stim {trial['stim'][-10:]}_x.csv", list(zip(trial['timestep'], trial['x'])), delimiter=',')
    #    np.savetxt(f"ETRA_500Hz_to_30Hz_Downsample_wo_interpol_subj {trial['subj']}_stim {trial['stim'][-10:]}_y.csv", list(zip(trial['timestep'], trial['y'])), delimiter=',')
    #    exit()
    #########

    ########calculate new sampling freq.
    '''
    print(old_hz)
    df = pd.DataFrame(new_timesteps, columns= ['vals'])            
    temp = df[df.columns[0]]
    diff = temp.diff().mean() 
    newHz = (1/diff) * 1000
    print(newHz)
    '''
    return trial

#downsample with interpol
#1000Hz -> 30Hz ->  0. 33.33333333 66.66666667 100. 133.33 ... 2000.
def downsample(trial, new_hz, old_hz):
    step = 1000/new_hz                                     
    #FIFA: max_timestep = trial['timestep'][-1]
    max_timestep = trial['timestep'].iloc[-1]
    new_timesteps = np.arange(0, max_timestep, step)
    
    interpol_values_x = interp1d(trial.timestep, trial.x.reshape(1, -1), kind='cubic')(new_timesteps).reshape(-1)
    interpol_values_y = interp1d(trial.timestep, trial.y.reshape(1, -1), kind='cubic')(new_timesteps).reshape(-1)

    trial.timestep = new_timesteps                                                        
    trial.x = interpol_values_x                                                         
    trial.y = interpol_values_y

    #########For the comparison, type(trial['x'] -> np array
    if trial['subj'] == '062' and trial['stim'] == 'WALDO/wal003.bmp':
        np.savetxt(f"ETRA_500Hz_to_30Hz_Downsample_w_interpol_cubic_subj {trial['subj']}_stim {trial['stim'][-10:]}_x.csv", list(zip(trial['timestep'], trial['x'])), delimiter=',')
        np.savetxt(f"ETRA_500Hz_to_30Hz_Downsample_w_interpol_cubic_subj {trial['subj']}_stim {trial['stim'][-10:]}_y.csv", list(zip(trial['timestep'], trial['y'])), delimiter=',')
        exit()
    #####################
    
    return trial

# upsample interpol
def upsample(trial, new_hz, old_hz):
    #print('In upsample'.upper())

    ########################################for comparison
    folder1 = 'Downsample_Comparison'
    folder2 = 'ETRA_500Hz_subj 009_stim nat004.bmp'
    filex = 'ETRA_500Hz_to_30Hz_Downsample_w_interpol_linear_subj 009_stim nat004.bmp_x.csv'
    filey = 'ETRA_500Hz_to_30Hz_Downsample_w_interpol_linear_subj 009_stim nat004.bmp_y.csv'
    x = np.loadtxt(f"{folder1}/{folder2}/{filex}", delimiter=',')
    x = np.transpose(x)
    trial.x = x[1]
    #print(trial.x[:10])
    y = np.loadtxt(f"{folder1}/{folder2}/{filey}", delimiter=',')
    y = np.transpose(y)
    trial.y = y[1]
    #print(trial.y[:10])
    trial.timestep = x[0]
    new_hz = 500               #!WICHTIG
    ########################################

    step = 1000/new_hz                                     
    max_timestep = trial['timestep'][-1]
    new_timesteps = np.arange(0, max_timestep, step)

    interpol_values_x = interp1d(trial.timestep, trial.x.reshape(1, -1), kind='cubic')(new_timesteps).reshape(-1)
    interpol_values_y = interp1d(trial.timestep, trial.y.reshape(1, -1), kind='cubic')(new_timesteps).reshape(-1)
    # 1000Hz -> 1200Hz: 0    1    2 ... 2018 2019 2020 -> 0.00000000e+00 8.33333333e-01 1.66666667e+00 ... 2.01750000e+03 2.01833333e+03 2.01916667e+03
    # length: 2021 -> 2424
    trial.timestep = new_timesteps                                       
    trial.x = interpol_values_x                                                         #(2877,) 
    trial.y = interpol_values_y  
    # trial.timestep[40] = 33.33 -> value: 0.47371575 -> old value at timestep 33.33 = 0.47320688
    # last value new: 0.76853863, old: 0.76855228    
    
    np.savetxt(f"ETRA_30Hz_wIntLin_to_500Hz_Upsample_w_interpol_cubic_subj 009_stim nat004.bmp_x.csv", list(zip(trial['timestep'], trial['x'])), delimiter=',')
    np.savetxt(f"ETRA_30Hz_wIntLin_to_500Hz_Upsample_w_interpol_cubic_subj 009_stim nat004.bmp_y.csv", list(zip(trial['timestep'], trial['y'])), delimiter=',')
     
    
    '''
    #print(old_hz)
    df = pd.DataFrame(new_timesteps, columns= ['vals'])            
    temp = df[df.columns[0]]
    diff = temp.diff().mean() 
    newHz = (1/diff) * 1000
    print(newHz)
    '''
    exit()
    
    return trial

#19570375   19570382 -> new_points = ...376, ...377, ..., ...381
def upsample_between_timestamp_pairs(trial, new_hz, old_hz, new_points, step):
    between_points = []
    # (19570375   19570382)
    for pointpair in new_points:
        start = pointpair[0]
        end = pointpair[1]
        for i in range(start+step, end, step): 
            between_points.append(i)
    
    between_points = np.array(between_points, dtype=int)
    interpol_values_x = interp1d(trial.timestep, trial.x.reshape(1, -1), kind='cubic')(between_points).reshape(-1)
    interpol_values_y = interp1d(trial.timestep, trial.y.reshape(1, -1), kind='cubic')(between_points).reshape(-1)
    '''
    print('point pair ', pointpair)
    print('interpol x ', interpol_values_x)
    print('interpol y ', interpol_values_y)
    print(type(trial.x))
    print(trial.x.shape)
    print('trial x ',trial.x[trial.timestep == pointpair[0]], '...', trial.x[trial.timestep == pointpair[1]])
    print('trial y ',trial.y[trial.timestep == pointpair[0]], '...', trial.y[trial.timestep == pointpair[1]])
    '''
    
    # trial type = Series
    #print('BEFORE, length trial.timestep: ', len(trial.timestep))
    trial.timestep = np.append(trial.timestep, between_points)
    trial.x = np.append(trial.x, interpol_values_x)
    trial.y = np.append(trial.y, interpol_values_y)
    #print('AFTER, length trial.timestep: ', len(trial.timestep))
    #axes: [Index(['subj', 'stim', 'task', 'timestep', 'x', 'y'], dtype='object')]
    sort_indices = np.argsort(trial.timestep)
    trial.timestep = trial.timestep[sort_indices]
    trial.x = trial.x[sort_indices]
    trial.y = trial.y[sort_indices]                             #->[170094 170095 170096 170097 170098 170099 170100 170101 170102 170103 170104 170105 170106 170107 170108 170109]
                                                                #print(trial.timestep[1640:1656])
    #print('SORTED ', trial.timestep[-20:])
    #print('is trial timestep sorted AFTER sort', np.all(np.diff(trial.timestep) >= 0))
    print('###################################################################')
    return trial

    '''
    Example: 
    point pair: (170096, 170108)
    timestemp           x              y
    170096          0.90968252      0.
    ----------------------------------------------interpol points
    170097          0.91118736      0.0093068
    170098          0.90722695      0.02862736
    170099          0.89879121      0.05595665
    170100          0.88687004      0.08928966
    170101          0.87245337      0.12662136
    170102          0.85653111      0.16594673
    170103          0.84009315      0.20526075
    170104          0.82412942      0.24255839
    170105          0.80962983      0.27583462
    170106          0.79758428      0.30308444
    170107          0.78898269      0.32230281
    ----------------------------------------------interpol points
    170108          0.78481497      0.33148471


    Example: CH stim:  0117.jpg
    point pair: (1610, 1667)
    timestemp           x              y
    1610          0.59143851      0.72141541
    ----------------------------------------------interpol points
    1611          0.59408523      0.72651082
    1612          0.59634838      0.73035547
    ...

    1665         0.50041198       0.44834106
    1666         0.5008119        0.45845911
    ----------------------------------------------interpol points
    1667         0.5015264        0.47006027
    '''
    




