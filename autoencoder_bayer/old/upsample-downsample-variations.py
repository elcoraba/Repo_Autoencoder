#upsample:
#ETRA:
stim = 'wal003.bmp'
subj = '062'
folder1 = 'Downsample_Comparison'
folder2 = f'ETRA_500Hz_subj {subj}_stim {stim}.bmp'
fileL = f'ETRA_500Hz_to_30Hz_Downsample_w_interpol_linear_subj ID {subj}_stim {stim}.csv'
fileC = f'ETRA_500Hz_to_30Hz_Downsample_w_interpol_cubic_subj ID {subj}_stim {stim}.csv'
fileO = f'ETRA_500Hz_to_30Hz_Downsample_wo_interpol_subj ID {subj}_stim {stim}.csv'
#...
np.savetxt(f"ETRA_30Hz_wIntLin_to_500Hz_Upsample_w_interpol_cubic_subj ID {subj}_stim {stim}.csv", list(zip(trial['timestep'], trial['x'], trial['y'])), delimiter=',', header='t,x,y')
exit()

#FIFA:
stim = '0113.jpg'
subj = 'JV'
folder1 = 'Downsample_Comparison'
folder2 = f'FIFA_1000Hz_subj {subj}_stim {stim}'
fileL = f'FIFA_1000Hz_to_30Hz_Downsample_w_interpol_linear_subj ID {subj}_stim {stim}.csv'
fileC = f'FIFA_1000Hz_to_30Hz_Downsample_w_interpol_cubic_subj ID {subj}_stim {stim}.csv'
fileO = f'FIFA_1000Hz_to_30Hz_Downsample_wo_interpol_subj ID {subj}_stim {stim}.csv'
#...
np.savetxt(f"FIFA_30Hz_wIntLin_to_1000Hz_Upsample_w_interpol_linear_subj ID {subj}_stim {stim}.csv", list(zip(trial['timestep'], trial['x'], trial['y'])), delimiter=',', header='t,x,y', comments = '')
exit()
    
    