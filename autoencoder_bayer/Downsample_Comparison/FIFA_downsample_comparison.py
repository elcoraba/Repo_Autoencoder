#cd "C:/Users/sybil/Documents/Masterarbeit_2021_22/code/autoencoder_bayer/Downsample_Comparison"
import numpy as np
import matplotlib.pyplot as plt

class fifa_downsample_comparison:

    #value is either 'x' or 'y'
    #Hz30_wo: downsampling w/o interpol
    def draw(Hz1000, Hz30_wo, Hz30_w_cubic, Hz30_w_linear, value): #upsampledHz1000
        timestep_1000 = Hz1000[0]
        values_1000 = Hz1000[1]

        timestep_30 = Hz30_wo[0]
        values_30 = Hz30_wo[1]

        timestep_30_cubic = Hz30_w_cubic[0]
        values_30_cubic = Hz30_w_cubic[1]

        timestep_30_linear = Hz30_w_linear[0]
        values_30_linear = Hz30_w_linear[1]

        plt.figure(figsize=(10,4))
        plt.scatter(timestep_1000, values_1000, c = 'cornflowerblue', s = 0.5, label = '1000Hz')
        plt.scatter(timestep_30, values_30, c = 'silver', s=40, label = 'Downsampled to 30Hz, w/o interpol')
        plt.scatter(timestep_30_cubic, values_30_cubic, c = 'red', s=15, marker = 'd', label = 'Downsampled to 30Hz, w interpol - cubic')
        plt.scatter(timestep_30_linear, values_30_linear, c = 'skyblue', s=6, marker = '*', label = 'Downsampled to 30Hz, w interpol - linear')
        plt.title(f"{value}: FIFA_1000_to_30Hz")
        plt.xlabel('timestep')
        plt.ylabel(f"{value} values")
        plt.legend()
        plt.show()

    stim = '0113'
    subj = 'JV'
    ############## 0055
    #x_1000 = np.loadtxt(f"FIFA_1000Hz_subj {subj}_stim {stim}.jpg/FIFA_1000Hz_to_30Hz_Downsample_wo_interpol_subj {subj}_stim {stim}.jpg_withAdaptSamplFreq_x.csv", delimiter=',')
    #x_1000 = np.transpose(x_1000)
    ##############
    x_1000 = np.loadtxt(f"FIFA_1000Hz_subj {subj}_stim {stim}.jpg/FIFA_1000Hz_subj {subj}_stim {stim}.jpg_x.csv", delimiter=',')
    x_1000 = np.transpose(x_1000)
    x_30_wo = np.loadtxt(f"FIFA_1000Hz_subj {subj}_stim {stim}.jpg/FIFA_1000Hz_to_30Hz_Downsample_wo_interpol_subj {subj}_stim {stim}.jpg_x.csv", delimiter=',')
    x_30_wo = np.transpose(x_30_wo)
    x_30_w_cubic = np.loadtxt(f"FIFA_1000Hz_subj {subj}_stim {stim}.jpg/FIFA_1000Hz_to_30Hz_Downsample_w_interpol_cubic_subj {subj}_stim {stim}.jpg_x.csv", delimiter=',')
    x_30_w_cubic = np.transpose(x_30_w_cubic)
    x_30_w_linear = np.loadtxt(f"FIFA_1000Hz_subj {subj}_stim {stim}.jpg/FIFA_1000Hz_to_30Hz_Downsample_w_interpol_linear_subj {subj}_stim {stim}.jpg_x.csv", delimiter=',')
    x_30_w_linear = np.transpose(x_30_w_linear)
    draw(x_1000, x_30_wo, x_30_w_cubic, x_30_w_linear, 'x')
    ##############################
    y_1000 = np.loadtxt(f"FIFA_1000Hz_subj {subj}_stim {stim}.jpg/FIFA_1000Hz_subj {subj}_stim {stim}.jpg_y.csv", delimiter=',')
    y_1000 = np.transpose(y_1000)
    y_30_wo = np.loadtxt(f"FIFA_1000Hz_subj {subj}_stim {stim}.jpg/FIFA_1000Hz_to_30Hz_Downsample_wo_interpol_subj {subj}_stim {stim}.jpg_y.csv", delimiter=',')
    y_30_wo = np.transpose(y_30_wo)
    y_30_w_cubic = np.loadtxt(f"FIFA_1000Hz_subj {subj}_stim {stim}.jpg/FIFA_1000Hz_to_30Hz_Downsample_w_interpol_cubic_subj {subj}_stim {stim}.jpg_y.csv", delimiter=',')
    y_30_w_cubic = np.transpose(y_30_w_cubic)
    y_30_w_linear = np.loadtxt(f"FIFA_1000Hz_subj {subj}_stim {stim}.jpg/FIFA_1000Hz_to_30Hz_Downsample_w_interpol_linear_subj {subj}_stim {stim}.jpg_y.csv", delimiter=',')
    y_30_w_linear = np.transpose(y_30_w_linear)
    draw(y_1000, y_30_wo, y_30_w_cubic, y_30_w_linear, 'y')







