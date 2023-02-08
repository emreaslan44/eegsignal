
from mne.filter import filter_data

def filter_InData(X, fs = 256,l_freq1 = 2, h_freq1 = 30):
  """band pass iir filter """
  filtered_eeg = filter_data(data = X,sfreq = fs,l_freq = l_freq1, h_freq = h_freq1, method='iir',verbose=0)
  return filtered_eeg