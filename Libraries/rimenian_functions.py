import numpy as np


from pyriemann.utils.base import invsqrtm
def parallel_transport_covariance_matrix(C, R):
    return np.dot(invsqrtm(R), np.dot(C, invsqrtm(R)))

"""Recentering of Data"""
def recenter_data(covs, Labels, transform_point, verbose=True):
    temp = []
    for Ci in covs:
        temp.append(parallel_transport_covariance_matrix(Ci,transform_point))
    data_rct = {}
    data_rct['covs'] = np.stack(temp)
    data_rct['labels'] = Labels
    if verbose:
        print(data_rct['covs'].shape)
    return data_rct['covs'], data_rct['labels']



"""Covarience Matrix Function"""
def get_conv_matrix(X, Label, epoch_start = 0, epoch_duration = 1):
  from mne.filter import filter_data
  from pyriemann.estimation import Covariances

  fs = 256
  #filtered_eeg = filter_data(data = X,sfreq = fs,l_freq = 2.0, h_freq = 30.0, method='iir',verbose=0)
  filtered_eeg = X
  epoch_end = epoch_start + epoch_duration
  epoch = filtered_eeg[:,:,int(epoch_start*fs):int(epoch_end*fs+1)]
  covs = Covariances().transform(X = epoch)

  from pyriemann.utils.mean import mean_riemann
  mean_source = mean_riemann(covs)

  #Label = np.concatenate((a1, a1), axis=0, out=None, dtype=None, casting="same_kind")

  source_rct_cov, source_rct_label = recenter_data(covs = covs, Labels = Label, transform_point=mean_source)

  return source_rct_cov, source_rct_label
