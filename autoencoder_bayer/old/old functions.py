#old functions

# old downsample with interpol
'''
def downsample(trial, new_hz, old_hz):
    factor = (new_hz / old_hz)
    num_downsampled_points = len(trial.x) * factor                                    
    new_trial = np.arange(0, num_downsampled_points, factor)                #MIT: 250->125, [  0.  0.52083333   1.04166667   1.5625  2.08333333 2.60416667   3.125 ... 373.95833333 374.47916667]
    new_points = np.arange(0, num_downsampled_points - 1, 1) 
    if len(new_trial) == len(trial.x) + 1 :
        
        if int(num_downsampled_points) != 404:
            print('Cheat')
            print('factor ', factor)
            print('num downsampled points ', num_downsampled_points) 
            print('new trial ', new_trial.shape)
            print('new points ', new_points.shape)
            print('trial shape ', trial.x.shape)
            print('new trial ', new_trial)
            print('new points ', new_points[0], ' ', new_points[-1])
        
        new_trial = new_trial[:-1]

    interpol_values_x = interp1d(new_trial, trial.x.reshape(1, -1), kind='cubic')(new_points).reshape(-1)
    interpol_values_y = interp1d(new_trial, trial.y.reshape(1, -1), kind='cubic')(new_points).reshape(-1)

    trial.timestep = new_points
    trial.x = interpol_values_x 
    trial.y = interpol_values_y

    return trial

# upsample with interpol OLD VERSION
def upsample(trial, new_hz, old_hz):
    #print('In upsample'.upper())
    factor = int(new_hz / old_hz)
    #-----------------old version
    
    #num_upsampled_points = len(trial.x) * factor    #2880 = 720 * 4
    #points = np.arange(0, num_upsampled_points, factor) #(720,)
    #new_points = np.arange(0, num_upsampled_points - (factor - 1), 1)
    #print('points shape ', new_points) # (2877,) [   0    1    2 ... 2874 2875 2876]
    #a = interp1d(points, trial.x.reshape(1, -1), kind='cubic')(new_points).reshape(-1)
    #b = interp1d(points, trial.y.reshape(1, -1), kind='cubic')(new_points).reshape(-1)
    #print('OLD interpol x : ', a[-30:])
    #print('OLD interpol y : ', b[-30:])
    #print('OLD points: ', points[-30:])
    
    #-----------------
    num_upsampled_points = len(trial.x) * factor                                        #TODO MIT: mit trial.timestep gibt es zwischendurch Problem, dann eventuell auch in downsample ändern?
    new_trial = np.arange(0, num_upsampled_points, factor)
    new_points = np.arange(0, num_upsampled_points - (factor - 1), 1)
    interpol_values_x = interp1d(new_trial, trial.x.reshape(1, -1), kind='cubic')(new_points).reshape(-1)
    interpol_values_y = interp1d(new_trial, trial.y.reshape(1, -1), kind='cubic')(new_points).reshape(-1)

    trial.timestep = new_points                                                         #MIT: # 0 ... 719 -> 0 ... 2876 

    trial.x = interpol_values_x                                                         #(2877,) 
    trial.y = interpol_values_y                 

    # old trial.x[0:5]: [0.44242484 0.44024921 0.44024921 0.44024921 0.43807358]
    # NEW interpol x:  [0.44242484 0.44161009 0.44098864 0.44054138 
    # 0.44024921 0.44009301 0.44005368 0.44011212 
    # 0.44024921 0.44043096 0.44056383 0.44053939
    # 0.44024921 0.4396444  0.43891428 0.43830772 
    # 0.43807358 0.43837338
    
    #print('NEW interpol x: ', trial.x[-30:])
    #print('NEW interpol y: ', trial.y[-30:])
    #print('NEW interpol timeste: ', trial.timestep[-30:])
    
    return trial

    #downsample without interpol
    #FIFA############################################################################
    def downsample_a(trial, new_hz, old_hz):
    #########For the comparison, type(trial['x'] -> np array
    #print('In downsample ', trial['subj'])
    #if trial['subj'] == 'CH' and trial['stim'] == '0055.jpg':
    #    np.savetxt(f"FIFA_1000Hz_subj {trial['subj']}_stim {trial['stim']}_x.csv", list(zip(trial['timestep'], trial['x'])), delimiter=',')
    #    np.savetxt(f"FIFA_1000Hz_subj {trial['subj']}_stim {trial['stim']}_y.csv", list(zip(trial['timestep'], trial['y'])), delimiter=',')
    #########
    ##ETRA:
    #if trial['subj'] == '022' and trial['stim'] == 'WALDO/wal004.bmp':
    #    np.savetxt(f"ETRA_500Hz_subj {trial['subj']}_stim {trial['stim'][-10:]}_x.csv", list(zip(trial['timestep'], trial['x'])), delimiter=',')
    #    np.savetxt(f"ETRA_500Hz_subj {trial['subj']}_stim {trial['stim'][-10:]}_y.csv", list(zip(trial['timestep'], trial['y'])), delimiter=',')
    ####
    step = 1000/new_hz                                      # hier darf man nicht runden! Sonst machen wir wieder nur 1, 2, ... Schritte
    max_timestep = trial['timestep'][-1]
    #ETRA: max_timestep = trial['timestep'].iloc[-1]
    new_timesteps = np.arange(0, max_timestep, step)
    new_timesteps = new_timesteps.round().astype(int)
 
    idx = np.intersect1d(trial.timestep, new_timesteps, return_indices=True)[1]                                                
    trial.timestep = trial.timestep[idx]
    trial.x = trial.x[idx]
    trial.y = trial.y[idx]

    #########For the comparison, type(trial['x'] -> np array
    #if trial['subj'] == 'CH' and trial['stim'] == '0055.jpg':
    #    np.savetxt(f"FIFA_1000Hz_to_30Hz_Downsample_wo_interpol_subj {trial['subj']}_stim {trial['stim']}_x.csv", list(zip(trial['timestep'], trial['x'])), delimiter=',')
    #    np.savetxt(f"FIFA_1000Hz_to_30Hz_Downsample_wo_interpol_subj {trial['subj']}_stim {trial['stim']}_y.csv", list(zip(trial['timestep'], trial['y'])), delimiter=',')
    #########
    ###ETRA:
    #NATURAL/nat014
    #022, PUZZLE/puz010.bmp
    #if trial['subj'] == '022' and trial['stim'] == 'WALDO/wal004.bmp':
    #    np.savetxt(f"ETRA_500Hz_to_30Hz_Downsample_wo_interpol_subj {trial['subj']}_stim {trial['stim'][-10:]}_x.csv", list(zip(trial['timestep'], trial['x'])), delimiter=',')
    #    np.savetxt(f"ETRA_500Hz_to_30Hz_Downsample_wo_interpol_subj {trial['subj']}_stim {trial['stim'][-10:]}_y.csv", list(zip(trial['timestep'], trial['y'])), delimiter=',')
    #    exit()
    ######

    return trial

    #downsample with interpol
    #1000Hz -> 30Hz ->  0. 33.33333333 66.66666667 100. 133.33 ... 2000.
    def downsample(trial, new_hz, old_hz):
    step = 1000/new_hz                                     
    max_timestep = trial['timestep'][-1]
    #ETRA: max_timestep = trial['timestep'].iloc[-1]
    new_timesteps = np.arange(0, max_timestep, step)
    
    interpol_values_x = interp1d(trial.timestep, trial.x.reshape(1, -1), kind='cubic')(new_timesteps).reshape(-1)
    interpol_values_y = interp1d(trial.timestep, trial.y.reshape(1, -1), kind='cubic')(new_timesteps).reshape(-1)

    trial.timestep = new_timesteps                                                        
    trial.x = interpol_values_x                                                         
    trial.y = interpol_values_y

    #########For the comparison, type(trial['x'] -> np array
    #if trial['subj'] == 'CH' and trial['stim'] == '0001.jpg':
    #    np.savetxt(f"FIFA_1000Hz_to_30Hz_Downsample_w_interpol_cubic_subj {trial['subj']}_stim {trial['stim']}_x.csv", list(zip(trial['timestep'], trial['x'])), delimiter=',')
    #    np.savetxt(f"FIFA_1000Hz_to_30Hz_Downsample_w_interpol_cubic_subj {trial['subj']}_stim {trial['stim']}_y.csv", list(zip(trial['timestep'], trial['y'])), delimiter=',')
    #    exit()
    #########
    ##ETRA: 
    if trial['subj'] == '062' and trial['stim'] == 'WALDO/wal003.bmp':
        np.savetxt(f"ETRA_500Hz_to_30Hz_Downsample_w_interpol_cubic_subj {trial['subj']}_stim {trial['stim'][-10:]}_x.csv", list(zip(trial['timestep'], trial['x'])), delimiter=',')
        np.savetxt(f"ETRA_500Hz_to_30Hz_Downsample_w_interpol_cubic_subj {trial['subj']}_stim {trial['stim'][-10:]}_y.csv", list(zip(trial['timestep'], trial['y'])), delimiter=',')
        exit()
    #####
    
    return trial

    # upsample interpol
    def upsample(trial, new_hz, old_hz):
    #print('In upsample'.upper())

    ########################################for comparison
    folder1 = 'Downsample_Comparison'
    folder2 = 'FIFA_1000Hz_subj JV_stim 0001.jpg'
    filex = 'FIFA_1000Hz_to_30Hz_Downsample_wo_interpol_subj JV_stim 0001.jpg_x.csv'
    filey = 'FIFA_1000Hz_to_30Hz_Downsample_wo_interpol_subj JV_stim 0001.jpg_y.csv'
    ##ETRA:
    folder1 = 'Downsample_Comparison'
    folder2 = 'ETRA_500Hz_subj 009_stim nat004.bmp'
    filex = 'ETRA_500Hz_to_30Hz_Downsample_w_interpol_linear_subj 009_stim nat004.bmp_x.csv'
    filey = 'ETRA_500Hz_to_30Hz_Downsample_w_interpol_linear_subj 009_stim nat004.bmp_y.csv'
    #####
    x = np.loadtxt(f"{folder1}/{folder2}/{filex}", delimiter=',')
    x = np.transpose(x)
    trial.x = x[1]
    #print(trial.x[:10])
    y = np.loadtxt(f"{folder1}/{folder2}/{filey}", delimiter=',')
    y = np.transpose(y)
    trial.y = y[1]
    #print(trial.y[:10])
    trial.timestep = x[0]
    new_hz = 1000               #!WICHTIG
    #ETRA: new_hz = 500               #!WICHTIG
    ########################################

    step = 1000/new_hz                                     
    max_timestep = trial['timestep'][-1] #FIFA
    #ETRA: max_timestep = trial['timestep'][-1]
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
    
    np.savetxt(f"FIFA_30Hz_woInt_to_1000Hz_Upsample_w_interpol_cubic_subj JV_stim 0001.jpg_x.csv", list(zip(trial['timestep'], trial['x'])), delimiter=',')
    np.savetxt(f"FIFA_30Hz_woInt_to_1000Hz_Upsample_w_interpol_cubic_subj JV_stim 0001.jpg_y.csv", list(zip(trial['timestep'], trial['y'])), delimiter=',')
    ##ETRA:
    np.savetxt(f"ETRA_30Hz_wIntLin_to_500Hz_Upsample_w_interpol_cubic_subj 009_stim nat004.bmp_x.csv", list(zip(trial['timestep'], trial['x'])), delimiter=',')
    np.savetxt(f"ETRA_30Hz_wIntLin_to_500Hz_Upsample_w_interpol_cubic_subj 009_stim nat004.bmp_y.csv", list(zip(trial['timestep'], trial['y'])), delimiter=',')
    #####
    exit() 
    
    return trial
    
    #FIFA############################################################################


    # was in corpusData
    def mean_duration_between_samples(self, values):
        #print(type(values.values))  #numpy.ndarray
        if type(values) == np.ndarray:
            df = pd.DataFrame(values, columns= ['vals'])            #FIFA
        else:
            df = pd.DataFrame(values.values, columns= ['vals'])     #ETRA
        #print('DATAFRAME: ', '\n', df)
        temp = df[df.columns[0]]
        diff = temp.diff().mean()
        
        #print('DIFF: ', diff)      #Always 2 for ETRA
        return diff


    def sampling_rate(self):
        #print('Calculate samling freq. manually')
        for subj in self.time:
            for trial, values in self.time[subj].items():
                meanTimeBetweenSamples = self.mean_duration_between_samples(values)
                timeBetweenSamples = (1/meanTimeBetweenSamples) * 1000
                if timeBetweenSamples != self.hz:
                    print('The calculated sampling frequency in subject ', subj, 'and trial ', trial, 'differs from the given frequency ', self.hz)
                    print('Calculated sampling frequency: ', timeBetweenSamples)


#######################old train loop
def train_old(self, args, run_identifier):
        print('################################# Start Training')
        logging.info('\n===== STARTING TRAINING =====')
        logging.info('{} samples, {} batches.'.format(
                     len(self.dataset), len(self.dataloader)))
        logging.info('Loss Fn:' + str(self.loss_fn))

        _checkpoint_interval = len(self.dataloader)
        num_checkpoints = int(MAX_TRAIN_ITERS / _checkpoint_interval)           #Anzahl der checkpoints
        self.init_global_losses(num_checkpoints + 1)

        i, e = 0, 0
        _checkpoint_start = time.time()
        while i < MAX_TRAIN_ITERS: #  20000
            #print('############ i: ', i , '/', MAX_TRAIN_ITERS)
            self.reset_epoch_losses()

            for b, batch in enumerate(self.dataloader):
                print(i, end = ' ')
                self.model.network.train()
                sample, sample_rec = self.forward(batch)
                
                i += 1
                if i % _checkpoint_interval == 0:
                    self.update_global_losses(int(i / _checkpoint_interval) - 1)
                    self.log(i, _checkpoint_interval,
                             time.time() - _checkpoint_start)
                    self.reset_epoch_losses()

                    if (e + 1) % 10 == 0:
                        #self.evaluate_representation(sample, sample_rec, i)
                        if self.save_model:
                            self.model.save(i, run_identifier, self.global_losses, args)

                    _checkpoint_start = time.time()
            print('loss per epoch ', self.epoch_losses)
            e += 1

'''