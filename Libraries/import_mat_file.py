"""
Note if you wanna import .mat files, .mat files should be saved latest version of matlab and its 
"""
from mat4py import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# print(len(lst["y"]))
def import_channels(mat_file_path, ch_array, sbj):
    """Load .mat file which contains sbj_x_ss_x data which has 15 channel data with size of 23809
    sbj_x_ss_x: determines the subject x and session x >> [x's are between 1 and 4]
    """
    lst = loadmat(mat_file_path)
    for ind in range(15):
        dlt_vec = 256*3 + 1
        del lst[sbj][ind][0:dlt_vec] 
    """Delete first 3 second of data """
    newData = list(zip(lst[sbj][0], lst[sbj][1], lst[sbj][2], lst[sbj][3], lst[sbj][4], lst[sbj][5], lst[sbj][6], lst[sbj][7], lst[sbj][8], lst[sbj][9], lst[sbj][10], lst[sbj][11], lst[sbj][12], lst[sbj][13], lst[sbj][14]))
    columns = ['time', 'CH0', 'CH1', 'CH2', 'CH3', 'CH4', 'CH5','CH6', 'CH7', 'CH8','CH9', 'CH10', 'CH11','CH12', 'CH13']
    """Create Dataframe and save as dataframe"""
    df = pd.DataFrame(newData, columns=columns)
    df_new = df[ch_array]
    """Select requested channels from dataframe"""
    return df_new

#Example of Function


# df_ch = import_channels(mat_file_path =r'C:\Users\yasar\Masaüstü\SDP_Final_Codes\eyeopenclose\eye_open_close.mat', ch_array=['CH0', 'CH1', 'CH3', 'CH4'], sbj = "sbj_2_ss_1")
# print(df_ch.to_numpy().shape)
# plt.plot(df_ch.to_numpy())
# plt.show()


