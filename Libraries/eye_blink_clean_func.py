"""Critical: add Library modules in here"""
from freq2scale_func import *
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy.signal import find_peaks
from statistics import median
import numpy as np



def cluster_close_numbers(numbers, threshold):
    clusters = []
    current_cluster = []
    for i, n in enumerate(numbers):
        if i == 0:
            current_cluster.append(n)
        else:
            if abs(n - numbers[i - 1]) <= threshold:
                current_cluster.append(n)
            else:
                clusters.append(current_cluster)
                current_cluster = [n]
    clusters.append(current_cluster)
    return clusters


def cleanEEG(path,fs,wavelet_type = 'morl',threshold_sec = 2, plotting_stat = True, start_signal = 0, end_signal=-1,  sensivity = 1000, deleteNumberofSample = int):
   
    ### arrangement of the data to get accurate format
    data = pd.read_csv(path, 
                    skiprows=0 ) 

    print(data.shape)

    if(data.shape[0] >data.shape[1] ):
        data = np.transpose(data)
        channel_numb = data.shape[0]
    else:
        channel_numb = data.shape[1]


    df = pd.DataFrame(data)
    ## continues wavelet transform and obtaining its frequencies 
    wavelet = pywt.ContinuousWavelet(wavelet_type)
    frequencies = np.array([10.5375389584087,9.83187149779973,9.17346047599755,8.55914127066466,7.98596118475440,7.45116525450823,6.95218300784893,6.48661610952464,6.05222683362063,5.64692730803239,5.26876947920435,4.91593574889987,4.58673023799878,4.27957063533233,3.99298059237720,3.72558262725411,3.47609150392446,3.24330805476232,3.02611341681031,2.82346365401620]) / fs # normalize
    scales = frequency2scale(wavelet,frequencies)
    dftoNP = df.to_numpy().T
    transformer = FastICA(n_components=channel_numb,  ## installing FastICA
            whiten='unit-variance')
    ics = transformer.fit_transform(dftoNP)  ## applying ICA to the data and getting independent components

    ics_new = ics.T
    ics_new_trans = ics
    peak_locs = []


    for channel in range(0,channel_numb):
        ics_square = np.power(ics_new_trans[:,channel],2)
        cwtmatr5, freqs = pywt.cwt(ics_new_trans[:,channel],scales, wavelet_type ,sampling_period=1/fs)
        cwtmatr6, freqs = pywt.cwt(ics_square,scales, wavelet_type,sampling_period=1/fs)
        multiplyResult = np.multiply(np.abs(cwtmatr5),np.abs(cwtmatr6))
        w_coef = np.sum(np.power(multiplyResult,2),axis=0)/20   
     
        locss_Amp = dftoNP[:,channel]
        median_Val = median(w_coef)*sensivity
        peaks, w = find_peaks(w_coef, height=[median_Val],distance=(fs/5))  ## getting peaks 

        clustered_peaks = cluster_close_numbers(peaks,int(threshold_sec*fs)) ## obtaining neighbouring peaks to chosee one of them
        address_mat= []

        ## chosing maximum peak among neioghboruing ones
        for cluster in clustered_peaks:
            val_peak = 0
            counter = len(cluster)
            for numb in cluster:
                if(w_coef [numb]> val_peak):
                
                    if len(address_mat) == 0:
                        pass
                    else:
                        if(val_peak > 0):
                            address_mat.pop()             
                    address_mat.append(numb)
                    val_peak = w_coef [numb]
                prev_cluster = cluster
        
        peaks = address_mat
        print("peaks, channel number: " + str(channel))
        peak_locs =  peak_locs + peaks
        print(np.array(peaks)/256)
        
        ## deleting samples before and after the peak
        if(len(peaks)> 0):
            deleteArea = np.array([])
            particularArea = []
            for i in range(0,len(peaks)):
                if(peaks[i]+ int(deleteNumberofSample)> len(ics_new_trans)):
                    upLimit = len(ics_new_trans)
                else:
                    upLimit = peaks[i]+ int(deleteNumberofSample)
                particularArea = list(range(peaks[i]- int(deleteNumberofSample), upLimit))  ## particularArea = list(range(peaks[i]- 50, upLimit))
                deleteArea = np.append(deleteArea,particularArea)


            clean_Pos = np.ones(len(dftoNP[:,channel]))
            for i in deleteArea:
                clean_Pos[int(i)] = 0

            temp = np.multiply(ics_new_trans[:,channel],clean_Pos)
            ics_new[channel,:] = temp

    ## obtaning clean EEG signal by doing matrix multiplication between mixing matrix of the ICA and new ICS's
    clean_eeg = np.transpose((np.matmul(transformer.mixing_,ics_new)))

    ## plotting hold on graphs of the cleand EEG's and non-clean EEG's 
    if(plotting_stat == True):     
        counter = 0
        channelNumb_half = int(math.ceil(channel_numb/2)) 
        fig, axs = plt.subplots(channelNumb_half,2,sharex="col")
        for i in range(0,channelNumb_half):
            axs[i,0].plot(np.arange(0,len( dftoNP[start_signal:end_signal,channel])/256,1/fs),dftoNP[start_signal:end_signal,counter],'tab:blue',np.arange(0,len( dftoNP[start_signal:end_signal,channel])/256,1/fs),clean_eeg[start_signal:end_signal,counter],'tab:red')
            axs[i,0].set_title('Channel ' + str(counter))
            if(i+1 !=  channel_numb-1):
                axs[i,1].plot(np.arange(0,len( dftoNP[start_signal:end_signal,channel])/256,1/fs),dftoNP[start_signal:end_signal,counter + 1],'tab:blue',np.arange(0,len( dftoNP[start_signal:end_signal,channel])/256,1/fs),clean_eeg[start_signal:end_signal,counter +1],'tab:red')
                axs[i,1].set_title('Channel ' + str(counter + 1))
            counter = counter + 2
     
        font = {'family' : 'normal',
                    'weight' : 'normal',
                    'size'   : 10}

        plt.rc('font', **font)
        fig.text(0.5, 0.04, 'Time(secs)', ha='center', va='center')
        fig.text(0.06, 0.5, 'Amplitude', ha='center', va='center', rotation='vertical')
        plt.show()
    print(peak_locs)
    arrranged_peak_locs = []
    counter_peaks = 0
    peak_mat = cluster_close_numbers(sorted(peak_locs),512)
    print(peak_mat)
    for sample_loc in peak_mat :
        arrranged_peak_locs.append(sample_loc[0])
        counter_peaks = counter_peaks + 1
    arrranged_peak_locs = np.array(arrranged_peak_locs)
    print(arrranged_peak_locs)


    
    return clean_eeg, peak_locs


#Code Example 

# path = r'C:\Users\emrre\OneDrive - agu.edu.tr\Masaüstü\SemesterProject\NewStartML_Brux\eyeBlinking_new1.csv' ## path of the data
# #example data şeklinde olsun sample adı
# fs_val= 256  # sampling rate
# deleteNumberofSample_val = int(fs_val/2)  # how many samples before and after peak should be deleted (saniye cisinden)
# wavelet_type = 'morl'  #type of the wavelet 
# threshold_Cluster = 3  #  neighbouring peaks second threshold
# plotting_status = True # enabling plotting
# sigStartSample= 0  # start sample for plotting
# sigEndSample=-1   #end sample for plotting
# sensitivityVal = 1000  # sensitivity value between 500 to 10000 for finding peaks

# cleanEEG(path,fs= fs_val, wavelet_type = wavelet_type, threshold_sec = threshold_Cluster, plotting_stat = plotting_status, start_signal = sigStartSample, end_signal=sigEndSample,sensivity =sensitivityVal, deleteNumberofSample =deleteNumberofSample_val )


# ##clean eeg fonksiyonun default olarak run edilmesi