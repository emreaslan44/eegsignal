import numpy as np

def data_formating(df, Fs= 256, Ch = 4):
    """ Converting EEG data, in format  n_epochs x n_channels * n_times to n_epochs x n_channels x n_times
      Example input Data: 90x256*4 === Converted EEG data format: 90x4x256
    """
    duration = int(len(df)/Fs)
    newdf = np.empty((duration,4))
    data = newdf
    once =True
    for sheet in range(Fs):
        for i in range(Ch): 
            for t in range(duration):
                ch = df[:,i]
                selection = np.arange(duration)
                selection = selection*Fs+sheet
                newdf[:,i] = ch[selection]
        data = np.dstack((data, newdf))
    data = np.delete(data, 0, axis=2)
    return data