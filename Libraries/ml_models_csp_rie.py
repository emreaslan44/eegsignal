from pyriemann.classification import MDM
import numpy as np
from pyriemann.estimation import Covariances
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from random import randrange
from sklearn import linear_model
from pyriemann.tangentspace import TangentSpace
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from pyriemann.spatialfilters import CSP
from sklearn.model_selection import cross_val_score, KFold

def multi_ml_model_train_test(indata, inlabel, train_cov):
    """
    indata: data with size of: 90x4x256 = sample_time x channels x Sampling_freq
    inlabel: data label with size of time vector
    train_cov: Covarience matrix 
    """
    ts = TangentSpace()
    ldac = LDA()
    lrc = LR()
    svc = SVC(kernel='linear')
    lasso = Lasso(alpha=0.5, fit_intercept=False)
    dtc = DecisionTreeClassifier()
    KNN = KNeighborsClassifier() 
    rfc = RandomForestClassifier(n_estimators=100)
    lrc = LR()


    accuracy_matrix  = []
    clf = make_pipeline(Covariances(), CSP(4), LDA(shrinkage='auto', solver='eigen'))
    acc = []
    for i in range(20):
        # Cross validation
        cv = KFold(n_splits=10, shuffle=True, random_state=randrange(10000))
        scores = cross_val_score(clf, indata, inlabel, cv=cv, scoring='accuracy',n_jobs=-1)
        acc.append(scores.mean())
    acc = np.array(acc)
    accuracy_matrix.append(acc.mean()) 


    clf = make_pipeline(Covariances(), CSP(4, log=False), TangentSpace(), lrc)
    acc = []
    for i in range(20):
    # Cross validation
        cv = KFold(n_splits=10, shuffle=True, random_state=randrange(10000))
        scores = cross_val_score(clf, indata, inlabel, cv=cv, scoring='accuracy',n_jobs=-1)
        acc.append(scores.mean())
    acc = np.array(acc)
    accuracy_matrix.append(acc.mean()) 

    accuracy_matrix  = []
    #add mdm
    clf = make_pipeline(ts, ldac)
    acc = []
    for i in range(20):
        # Cross validation
        cv = KFold(n_splits=10, shuffle=True, random_state=randrange(10000))
        scores = cross_val_score(clf, train_cov, inlabel, cv=cv)
        acc.append(scores.mean())
    acc = np.array(acc)
    accuracy_matrix.append(acc.mean())


    clf = make_pipeline(ts, lrc)
    acc = []
    for i in range(20):
        # Cross validation
        cv = KFold(n_splits=10, shuffle=True, random_state=randrange(10000))
        scores = cross_val_score(clf, train_cov, inlabel, cv=cv)
        acc.append(scores.mean())
    acc = np.array(acc)
    accuracy_matrix.append(acc.mean())


    clf = make_pipeline(ts, svc)
    acc = []
    for i in range(20):
        # Cross validation
        cv = KFold(n_splits=10, shuffle=True, random_state=randrange(10000))
        scores = cross_val_score(clf, train_cov, inlabel, cv=cv)
        acc.append(scores.mean())
    acc = np.array(acc)
    accuracy_matrix.append(acc.mean())


    clf = make_pipeline(ts, lasso)
    acc = []
    for i in range(20):
        # Cross validation
        cv = KFold(n_splits=10, shuffle=True, random_state=randrange(10000))
        scores = cross_val_score(clf, train_cov, inlabel, cv=cv, scoring='accuracy')
        acc.append(scores.mean())
    acc = np.array(acc)
    accuracy_matrix.append(acc.mean())


    clf = MDM()
    acc = []
    for i in range(20):
        # Cross validation
        cv = KFold(n_splits=10, shuffle=True, random_state=randrange(10000))
        scores = cross_val_score(clf, train_cov, inlabel, cv=cv, scoring='accuracy')
        acc.append(scores.mean())
    acc = np.array(acc)
    accuracy_matrix.append(acc.mean())


    clf = make_pipeline(ts, dtc)
    acc = []
    for i in range(20):
        # Cross validation
        cv = KFold(n_splits=10, shuffle=True, random_state=randrange(10000))
        scores = cross_val_score(clf, train_cov, inlabel, cv=cv, scoring='accuracy')
        acc.append(scores.mean())
    acc = np.array(acc)
    accuracy_matrix.append(acc.mean())


    clf = make_pipeline(ts, KNN)
    acc = []
    for i in range(20):
        # Cross validation
        cv = KFold(n_splits=10, shuffle=True, random_state=randrange(10000))
        scores = cross_val_score(clf, train_cov, inlabel, cv=cv, scoring='accuracy')
        acc.append(scores.mean())
    acc = np.array(acc)
    accuracy_matrix.append(acc.mean())



    clf = make_pipeline(ts, rfc)
    acc = []
    for i in range(20):
        # Cross validation
        cv = KFold(n_splits=10, shuffle=True, random_state=randrange(10000))
        scores = cross_val_score(clf, train_cov, inlabel, cv=cv, scoring='accuracy')
        acc.append(scores.mean())
    acc = np.array(acc)
    accuracy_matrix.append(acc.mean())


    clf = make_pipeline(Covariances(), CSP(4), LDA(shrinkage='auto', solver='eigen'))
    acc = []
    for i in range(20):
        # Cross validation
        cv = KFold(n_splits=10, shuffle=True, random_state=randrange(10000))
        scores = cross_val_score(clf, indata, inlabel, cv=cv, scoring='accuracy',n_jobs=-1)
        acc.append(scores.mean())
    acc = np.array(acc)
    accuracy_matrix.append(acc.mean()) 


    clf = make_pipeline(Covariances(), TangentSpace(), lrc)
    acc = []
    for i in range(20):
        # Cross validation
        cv = KFold(n_splits=10, shuffle=True, random_state=randrange(10000))
        scores = cross_val_score(clf, indata, inlabel, cv=cv, scoring='accuracy',n_jobs=-1)
        acc.append(scores.mean())
    acc = np.array(acc)
    accuracy_matrix.append(acc.mean()) 

    clf = make_pipeline(Covariances(), MDM())
    acc = []
    for i in range(20):
        # Cross validation
        cv = KFold(n_splits=10, shuffle=True, random_state=randrange(10000))
        scores = cross_val_score(clf, indata, inlabel, cv=cv, scoring='accuracy',n_jobs=-1)
        acc.append(scores.mean())
    acc = np.array(acc)
    accuracy_matrix.append(acc.mean()) 

    return accuracy_matrix
