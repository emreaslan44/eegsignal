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
    def __init__(self) -> None:
        """not determined any initial value for function"""
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

    def bandpower(x, fs, fmin, fmax):
        """The inputs are determined as data and fs, The bandpower of given interval(fmin and fmax) calculated"""
        f, Pxx = scipy.signal.periodogram(x, fs=fs)
        ind_min = np.argmax(f > fmin) - 1
        ind_max = np.argmax(f > fmax) - 1
        return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

    def alfaBetaRatio(self, data):
        """The bandpower ratio calculated based on bp[8Hz-32Hz] / bp[12Hz-30Hz]"""
        fs = 256   
        ############### 
        f_low1  = 8       
        f_high1 = 12      
        f_low2  = 12      
        f_high2 = 30

        BPA = Features.bandpower(data, fs, f_low1, f_high1)
        BPB = Features.bandpower(data, fs, f_low2, f_high2)
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

    def jBandPowerFunc(self, data, fs, f_low, f_high):
        BP = Features.bandpower(data, fs, f_low, f_high)
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


class Bruxism(Features):
    def __init__(self) -> None:
        """The objective of this function to create dataframe which contains the multiple features for based on time sample(t = 90)
        this class will create final features and there is no need for label because it is test code so we need additional script to test ML model
        """
        pass

    def Featuresdf(self, data, Fs):
        time = len(data)/Fs # time definition as 90 sec
        segmentedArry = np.split(data, (time)) # segment data into 90 
        ##Window = segmentedArry[0][:]
        #time vector 
        shape = np.shape(segmentedArry)
        ones = np.zeros(shape[0])
        #Label identification
        for i in range(shape[0]):
            if i > 9 and i<20:
                ones[i] = 1
            elif i > 29 and i<40:
                ones[i] = 1
            elif i > 49 and i<60:
                ones[i] = 1

        label = ones
        "Coloums to store each value at dataframe"
        cols = ['Mean', 'AutoRegModel', 'AlfaPower', 'BetaPower' , 'DeltaPower', 'GammaPower','ThetaPower',
        'FirstDiff','HActivity','HComplex','HMobility','Kurtosis',
        'LogEEntropy','LogSSVar','Maximum','MeanCurveL','MeanEnergy',
        'MeanTaeEnergy','Median','Minimum','NormFirstDiff','NormSecondDiff',
        'RatioAB','RenyEntropy','Skewness','SecondDiff','StandartDev','ThalisEnt',"Var"]

        df2 = pd.DataFrame(columns=cols, index=range(int(time)))
        for i in range(shape[0]):
            # Dividing into
            Window = segmentedArry[i][:]
            # Feature Extraction 
            # Extract and save each feeatures into dataframe for one Channel
            df2.loc[i].Mean = Features.jArithmeticMean(self, Window)
            df2.loc[i].AutoRegModel = Features.jAutoRegressiveModel(self, Window)[0]
            df2.loc[i].AlfaPower = Features.jBandPowerFunc(self, Window, fs=256,f_low=8, f_high=12)
            df2.loc[i].BetaPower = Features.jBandPowerFunc(self, Window, fs=256,f_low=12, f_high=30)
            df2.loc[i].DeltaPower = Features.jBandPowerFunc(self, Window, fs=256,f_low=1, f_high=4)
            df2.loc[i].GammaPower = Features.jBandPowerFunc(self, Window, fs=256,f_low=30, f_high=64)
            df2.loc[i].ThetaPower = Features.jBandPowerFunc(self, Window, fs=256,f_low=4, f_high=8)
            df2.loc[i].FirstDiff = Features.NormFirstDiff(self, Window)[0]
            df2.loc[i].HActivity = Features.jHjorthActivity(self, Window)
            df2.loc[i].HComplex = Features.jHjorthMobility(self, Window)
            df2.loc[i].HMobility = Features.jHjorthMobility(self, Window)
            df2.loc[i].Kurtosis = Features.kurtosisD(self, Window)
            df2.loc[i].LogEEntropy = Features.jLogEnergyEntropy(self, Window)
            df2.loc[i].LogSSVar = Features.jLogRootSumOfSequentialVariation(self, Window)
            df2.loc[i].Maximum = Features.maxF(self, Window)
            df2.loc[i].MeanCurveL = Features.jMeanCurveLength(self, Window)
            df2.loc[i].MeanEnergy = Features.jMeanEnergy(self, Window)
            df2.loc[i].MeanTaeEnergy = Features.jMeanTeagerEnergy(self, Window)
            df2.loc[i].Median = Features.median(self, Window)
            df2.loc[i].Minimum = Features.minimum(self, Window)
            df2.loc[i].NormFirstDiff = Features.NormFirstDiff(self, Window)[1]
            df2.loc[i].NormSecondDiff = Features.NormSecondDiff(self, Window)[1]
            df2.loc[i].RatioAB = Features.alfaBetaRatio(self, Window)
            df2.loc[i].RenyEntropy = Features.RenyEntropy(self, Window)
            df2.loc[i].Skewness = Features.Skewnes(self, Window)
            df2.loc[i].SecondDiff = Features.NormSecondDiff(self, Window)[0]
            df2.loc[i].StandartDev = Features.StandartDev(self, Window)
            df2.loc[i].ThalisEnt = Features.TsallisEntropy(self, Window)
            df2.loc[i].Var = Features.variance(self, Window)
            # Label is not necessary for test ml part
            #df2.loc[i].Label = label[i]
        return df2

    def LabelAdd(self, data, Fs):
        """Creating Labels based on Fs and determined data Length"""
        time = len(data)/Fs
        segmentedArry = np.split(data, (time)) 
        shape = np.shape(segmentedArry)
        ones = np.zeros(shape[0])

        for i in range(shape[0]):
            if i > 7 and i<18:
                ones[i] = 1
            elif i > 27 and i<38:
                ones[i] = 1
            elif i > 47 and i<58:
                ones[i] = 1
        label = ones
        return label

    def FinalFeatures(self, df):
        """Feature Extraction for different Channels and concatinating them into dataframe"""
        df1 = Bruxism.Featuresdf(self, df["CH0"].to_numpy(), Fs =256)
        df2 = Bruxism.Featuresdf(self, df["CH1"].to_numpy(), Fs=256)
        df3 = Bruxism.Featuresdf(self, df["CH2"].to_numpy(), Fs= 256)
        df4 = Bruxism.Featuresdf(self, df["CH3"].to_numpy(), Fs=256)
        df1.columns = 'CH0' + df1.columns
        df2.columns = 'CH1' + df2.columns
        df3.columns = 'CH2' + df3.columns
        df4.columns = 'Ch3' + df4.columns

        frames = [df1, df2, df3, df4]

        df5 = pd.concat(frames, axis=1)
        df5 = df5.reset_index(drop=True)

        # # use to add label
        # lbl = Bruxism.LabelAdd(self, data, Fs = 256)
        # df5["Label"] = lbl
        return df5


    def MachineLearningModel(self, df):
        """This function will helps to read trained machine learning mode beforehand"""
        #y_test = df["Label"]
        X_test = df
        with open('RandomForestModel','rb') as f:
            mp = pickle.load(f)
        PredictedLabel = mp.predict(X_test)
        return PredictedLabel


# read untrained Csv file

#df = pd.read_csv("inputData.csv") # thid is for at same folder location
df = pd.read_csv("inputData.csv")
#Call Bruxism Class which contains necessary Feature Extraction functions and ML model
bruxism = Bruxism()
df5 = bruxism.FinalFeatures(df)

print(df5)
# it will just predicted labels 
y_pred = bruxism.MachineLearningModel(df5)


plt.plot(y_pred)
plt.title('Predicted Bruxism Label')
plt.xlabel('time(sec)')
plt.ylabel('Label (0|1)')
plt.show()



"This part will be used to save dataframe into csv"
# df5.to_csv("BruxFeatures3.csv")
# print('Process is over ')