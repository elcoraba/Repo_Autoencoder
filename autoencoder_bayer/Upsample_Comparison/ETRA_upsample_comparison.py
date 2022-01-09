#cd "C:/Users/sybil/Documents/Masterarbeit_2021_22/code/autoencoder_bayer/Upsample_Comparison"
import numpy as np
import matplotlib.pyplot as plt

class etra_upsample_comparison:

    #value is either 'x' or 'y'
    #Hz30_wo: downsampling w/o interpol
    def draw(Hz1000, Hz1000_cubic, Hz1000_linear, value):
        timestep_1000 = Hz1000[0]
        values_1000 = Hz1000[1]

        timestep_1000_c = Hz1000_cubic[0]
        values_1000_c = Hz1000_cubic[1]

        timestep_1000_linear = Hz1000_linear[0]
        values_1000_linear = Hz1000_linear[1]


        plt.figure(figsize=(10,4))
        plt.plot(timestep_1000, values_1000, c = 'cornflowerblue', label = '1000Hz')
        plt.plot(timestep_1000_c, values_1000_c, c = 'silver', label = 'Upsampled from 30 to 1000Hz, w interpol - cubic')
        plt.plot(timestep_1000_linear, values_1000_linear, c = 'red', label = 'Upsampled from 30 to 1000Hz, w interpol - linear')
        plt.title(f"{value}: ETRA_30_to_500Hz")
        plt.xlabel('timestep')
        plt.ylabel(f"{value} values")
        plt.legend()
        plt.show()

    stim = 'wal003'
    subj = '062'
    ############## With 30Hz without interpol
    
    x_1000 = np.loadtxt(f"ETRA_500Hz_subj {subj}_stim {stim}.bmp/ETRA_500Hz_subj {subj}_stim {stim}.bmp_x.csv", delimiter=',')
    x_1000 = np.transpose(x_1000)
    x_1000_c = np.loadtxt(f"ETRA_500Hz_subj {subj}_stim {stim}.bmp/ETRA_30Hz_woInt_to_500Hz_Upsample_w_interpol_cubic_subj {subj}_stim {stim}.bmp_x.csv", delimiter=',')
    x_1000_c = np.transpose(x_1000_c)
    x_1000_l = np.loadtxt(f"ETRA_500Hz_subj {subj}_stim {stim}.bmp/ETRA_30Hz_woInt_to_500Hz_Upsample_w_interpol_linear_subj {subj}_stim {stim}.bmp_x.csv", delimiter=',')
    x_1000_l = np.transpose(x_1000_l)

    draw(x_1000, x_1000_c, x_1000_l, 'x')
    ##############################
    y_1000 = np.loadtxt(f"ETRA_500Hz_subj {subj}_stim {stim}.bmp/ETRA_500Hz_subj {subj}_stim {stim}.bmp_y.csv", delimiter=',')
    y_1000 = np.transpose(y_1000)
    y_1000_c = np.loadtxt(f"ETRA_500Hz_subj {subj}_stim {stim}.bmp/ETRA_30Hz_woInt_to_500Hz_Upsample_w_interpol_cubic_subj {subj}_stim {stim}.bmp_y.csv", delimiter=',')
    y_1000_c = np.transpose(y_1000_c)
    y_1000_l = np.loadtxt(f"ETRA_500Hz_subj {subj}_stim {stim}.bmp/ETRA_30Hz_woInt_to_500Hz_Upsample_w_interpol_linear_subj {subj}_stim {stim}.bmp_y.csv", delimiter=',')
    y_1000_l = np.transpose(y_1000_l)

    draw(y_1000, y_1000_c, y_1000_l, 'y')
    

    ############## With 30Hz with interpol linear
    x_1000_c = np.loadtxt(f"ETRA_500Hz_subj {subj}_stim {stim}.bmp/ETRA_30Hz_wIntLin_to_500Hz_Upsample_w_interpol_cubic_subj {subj}_stim {stim}.bmp_x.csv", delimiter=',')
    x_1000_c = np.transpose(x_1000_c)
    x_1000_l = np.loadtxt(f"ETRA_500Hz_subj {subj}_stim {stim}.bmp/ETRA_30Hz_wIntLin_to_500Hz_Upsample_w_interpol_linear_subj {subj}_stim {stim}.bmp_x.csv", delimiter=',')
    x_1000_l = np.transpose(x_1000_l)

    draw(x_1000, x_1000_c, x_1000_l, 'x')
    ##############################
    y_1000_c = np.loadtxt(f"ETRA_500Hz_subj {subj}_stim {stim}.bmp/ETRA_30Hz_wIntLin_to_500Hz_Upsample_w_interpol_cubic_subj {subj}_stim {stim}.bmp_y.csv", delimiter=',')
    y_1000_c = np.transpose(y_1000_c)
    y_1000_l = np.loadtxt(f"ETRA_500Hz_subj {subj}_stim {stim}.bmp/ETRA_30Hz_wIntLin_to_500Hz_Upsample_w_interpol_linear_subj {subj}_stim {stim}.bmp_y.csv", delimiter=',')
    y_1000_l = np.transpose(y_1000_l)

    draw(y_1000, y_1000_c, y_1000_l, 'y')







