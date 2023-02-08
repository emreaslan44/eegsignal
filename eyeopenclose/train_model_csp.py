import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mne.filter import filter_data
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from pyriemann.spatialfilters import CSP
from pyriemann.estimation import Covariances
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pickle

df = pd.read_csv(r"C:\Users\yasar\Masaüstü\SDP_Backend\EyeOpenClose\adgEyeDeneme.csv")
data = df.to_numpy()


def filter_InData(X, fs = 256):
  """band pass iir filter """
  filtered_eeg = filter_data(data = X,sfreq = fs,l_freq = 2.0, h_freq = 30.0, method='iir',verbose=0)
  return filtered_eeg

def label_Generator(duration = 90):
  """Creating Label based on expetimental setup"""
  ones = np.zeros(duration)
  for i in range(duration):
    if i > 9 and i<20:
      ones[i] = 1
    elif i > 29 and i<40:
      ones[i] = 1
    elif i > 49 and i<60:
      ones[i] = 1
    elif i > 69 and i<80:
      ones[i] = 1
  a1 = ones
  Labels = np.concatenate((a1, a1, a1, a1), axis=0, out=None, dtype=None, casting="same_kind")
  return Labels

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


df= pd.read_csv(r"C:\Users\yasar\Masaüstü\SDP_Backend\EyeOpenClose\adgEyeDeneme.csv")
data = df.to_numpy()
print(type(data))
data = filter_InData(X = data, fs = 256)

print(type(data))

# formatted_data = data_formating(df = data, Fs= 256, Ch = 4)
# Labels = label_Generator(duration = 90)
# X_train = filter_InData(X = formatted_data, fs = 256)

# print(X_train.shape, Labels.shape)
# "make pipeline of CSP and LDA to train model and save it"
# model = make_pipeline(Covariances(), CSP(4), LDA(shrinkage='auto', solver='eigen'))
# model.fit(X_train, Labels)

# import pickle
# with open('ml_model','wb') as f:
#     pickle.dump(model, f)