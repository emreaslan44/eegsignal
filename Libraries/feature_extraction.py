#neccessary python libraries
from math import log2
from msilib.schema import Class
from xml.sax.handler import DTDHandler
import pandas as pd 
import numpy as np 
import scipy
import scipy.signal
from scipy.stats import skew
from spectrum import Criteria
from scipy.stats import kurtosis
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle
import matplotlib.pyplot as plt

class Features():
    """
    In this part, multiple feature extraction functions implemented. The necessary explanations of these functions will be given from EEG Feature Extraction Toolbox. 
    Reference Link: https://www.mathworks.com/matlabcentral/fileexchange/84235-eeg-feature-extraction-toolbox?s_tid=FX_rc1_behav 
    you can obtain detailed information from referred link
    This test_trained_ml_model.py should be at the same and folder directory with "RandomForestModel.h5" and "inputData.csv"
    
    """
    def __init__(self, Fs):
        """initial value determined as Fs"""
        self.Fs = Fs
        pass

    def median(self, data): 
        return np.median(data)

    def minimum(self, data):
        return np.min(data)

    def NormFirstDiff(self, data):
        len = np.size(data)
        Y = 0
        for x in range(len-1):
            Y = Y + np.abs(data[x+1] - data[x])

        FD = (1/(len - 1)) * Y 
        NFD = FD / np.std(data) 
        return FD, NFD 

    def NormSecondDiff(self, data):
        len = np.size(data)
        Y = 0
        for x in range(len-2):
            Y = Y + np.abs(data[x+2] - data[x])

        SD = (1/(len - 2)) * Y 
        NSD = SD / np.std(data) 
        return SD, NSD 

    def bandpower(x, Fs, fmin, fmax):
        """The inputs are determined as data and fs, The bandpower of given interval(fmin and fmax) calculated"""
        f, Pxx = scipy.signal.periodogram(x, fs=Fs)
        ind_min = np.argmax(f > fmin) - 1
        ind_max = np.argmax(f > fmax) - 1
        return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

    def alfaBetaRatio(self, data, Fs):
        """The bandpower ratio calculated based on bp[8Hz-32Hz] / bp[12Hz-30Hz]""" 
        ############### 
        f_low1  = 8       
        f_high1 = 12      
        f_low2  = 12      
        f_high2 = 30

        BPA = Features.bandpower(data, Fs, f_low1, f_high1)
        BPB = Features.bandpower(data, Fs, f_low2, f_high2)
        RBA = BPB / BPA
        return RBA


    def RenyEntropy(self, data):
        alpha = 2 
        P = np.square(data) / np.sum(np.square(data))
        Ent = np.square(P)
        ReEnt = (1/(1 - alpha)) * log2(np.sum(Ent)) 
        return ReEnt

    def ShanonEntropy(self, data):
        P = np.square(data) / np.sum(np.square(data))
        Ent = np.multiply(P ,np.log2(P))
        ShEn = -np.sum(Ent)
        return ShEn

    def Skewnes(self, data):
        return skew(data)

    def StandartDev(self, data):
        len = np.size(data)
        mu = np.mean(data)
        SD = np.sqrt((1/(len-1)) * np.sum(np.square(data - mu)))
        return SD

    def TsallisEntropy(self, data):
        alpha = 2
        C = np.square(data) / np.sum(np.square(data))
        Ent = np.square(C)
        TsEnt = (1/(alpha - 1)) * (1 - np.sum(Ent))
        return TsEnt

    def variance(self, data):
        len = np.size(data)
        mu = np.mean(data)
        Var = (1 / (len - 1)) * np.sum(np.square(data - mu)) 
        return Var

    def jArithmeticMean(self, data):
        return np.mean(data)

    def arbug(self, X, order, criteria=None):
        if order == 0.: 
            raise ValueError("order must be > 0")

        x = np.array(X)
        N = len(x)

        # Initialisation
        # ------ rho, den
        rho = sum(abs(x)**2.) / float(N)  # Eq 8.21 [Marple]_
        den = rho * 2. * N 

        # ---- criteria
        if criteria:
            crit = Criteria(name=criteria, N=N)
            crit.data = rho
            print(0, 'old criteria=',crit.old_data, 'new criteria=',crit.data, 'new_rho=', rho)

        #p =0
        a = np.zeros(0, dtype=complex)
        ref = np.zeros(0, dtype=complex)
        ef = x.astype(complex)
        eb = x.astype(complex)
        temp = 1.
        #   Main recursion
        
        for k in range(0, order):
            
            # calculate the next order reflection coefficient Eq 8.14 Marple
            num = sum([ef[j]*eb[j-1].conjugate() for j in range(k+1, N)])
            den = temp * den - abs(ef[k])**2 - abs(eb[N-1])**2  
            kp = -2. * num / den #eq 8.14
            
            temp = 1. - abs(kp)**2.
            new_rho = temp * rho 

            if criteria:
                print(k+1, 'old criteria=',crit.old_data, 'new criteria=',crit.data, 'new_rho=',new_rho)
                #k+1 because order goes from 1 to P whereas k starts at 0.
                status = crit(rho=temp*rho, k=k+1)
                if status is False:
                    #print 'should stop here-----------------', crit.data, crit.old_data 
                    break
            # this should be after the criteria
            rho = new_rho
            if rho <= 0:
                raise ValueError("Found an negative value (expected positive stricly) %s" % rho)

            a.resize(a.size+1)
            a[k] = kp
            if k == 0:
                for j in range(N-1, k, -1):     
                    save2 = ef[j]
                    ef[j] = save2 + kp * eb[j-1]          # Eq. (8.7)
                    eb[j] = eb[j-1] + kp.conjugate() *  save2
                
            else:
                # update the AR coeff
                khalf = (k+1)/2
                for j in range(0, khalf):   
                    ap = a[j] # previous value
                    a[j] = ap + kp * a[k-j-1].conjugate()      # Eq. (8.2)     
                    if j != k-j-1:
                        a[k-j-1] = a[k-j-1] + kp * ap.conjugate()    # Eq. (8.2)

                # update the prediction error
                for j in range(N-1, k, -1):
                    save2 = ef[j]
                    ef[j] = save2 + kp * eb[j-1]          # Eq. (8.7)
                    eb[j] = eb[j-1] + kp.conjugate() *  save2
                
            # save the reflection coefficient
            ref.resize(ref.size+1)        
            ref[k] = kp
            return a, rho, ref

    def jAutoRegressiveModel(self, data):
        order = 4
        Y = Features.arbug(self, data, order)
        AR = Y[1:order]
        return AR

    def jBandPowerFunc(self, data, Fs, f_low, f_high):
        BP = Features.bandpower(data, Fs, f_low, f_high)
        return BP

    def jHjorthActivity(self, data):
        sd = np.std(data)
        return np.square(sd)


    def jHjorthComplexity(self, X):
        ini_list1 = np.insert(X, 0, 0)
        diff_list = np.diff(ini_list1)
        diff_list2 = np.insert(diff_list, 0, 0)
        diff_list2 =  np.diff(diff_list2)

        #Standard deviation 
        sd0 = np.std(X)
        sd1 = np.std(diff_list)
        sd2 = np.std(diff_list2)

        return  (sd2/sd1) / (sd1 / sd0)


    def jHjorthMobility(self, X): 
        ini_list1 = np.insert(X, 0, 0)
        diff_list = np.diff(ini_list1)

        #Standard deviation 
        sd0 = np.std(X)
        sd1 = np.std(diff_list)
        return sd1/sd0

    def kurtosisD(self, X):
        return kurtosis(X,fisher=False)


    def jLogEnergyEntropy(self, X):
        return sum(np.log(np.power(X,2)))

    def jLogRootSumOfSequentialVariation(self, X):
        N = len(X)
        Y = np.zeros(N-1)

        for k in range(1,N):
            Y[k-1] = np.power(X[k] - X[k-1],2)
        return np.log10(np.sqrt(np.sum(Y)))

    def maxF(self, X): 
        return np.max(X)

    def jMeanCurveLength(self, X):
        N = len(X)
        Y = 0
        for k in range(1,N):
            Y = Y + np.abs(X[k] - X[k-1])
        return (1/N)* Y

    def jMeanEnergy(self, X):
        return  np.mean(np.power(X , 2))

    def jMeanTeagerEnergy(self, X):
        N = len(X)
        Y = 0
        for i in range(2,N):
            Y = Y + (np.power(X[i-1],2) - X[i] * X[i-2])
        return 1/N*Y
