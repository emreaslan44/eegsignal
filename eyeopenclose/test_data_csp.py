import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mne.filter import filter_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pickle


class Eye_OC_Detection():
  def filter_InData(self, X, fs = 256):
    """band pass iir filter """
    filtered_eeg = filter_data(data = X,sfreq = fs,l_freq = 2.0, h_freq = 30.0, method='iir',verbose=0)
    return filtered_eeg

  def data_formating(self, df, Fs= 256, Ch = 4):
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

  def ml_model_predict(self, In_data):
    """Use of trained machine learning model to predict labels"""
    with open('ml_model','rb') as f:
      mp = pickle.load(f)
    y_pred = mp.predict(In_data)
    return y_pred

# Read Dataframe
df= pd.read_csv(r"C:\Users\yasar\Masaüstü\SDP_Backend\EyeOpenClose\adgEyeDeneme4.csv")

# make the process
data = df.to_numpy()
eye_oc = Eye_OC_Detection()
formatted_data = eye_oc.data_formating(df = data, Fs= 256, Ch = 4)
X_test = eye_oc.filter_InData(X = formatted_data, fs = 256)
y_pred = eye_oc.ml_model_predict(X_test)

eyeBlinks = np.array([0, 9, 19, 29, 39, 49, 59, 69, 79, 89])

inx = y_pred[eyeBlinks[4]]
#inx = y_pred.index(eyeBlinks[1])
print(inx)

# plt.plot(y_pred)
# plt.title('Predicted Bruxism Label')
# plt.xlabel('time(sec)')
# plt.ylabel('Label (0|1)')
# plt.show()
# # model_final.evaluate(X_test, y_test)

