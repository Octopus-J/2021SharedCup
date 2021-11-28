import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import svm

def svm_pred(x,y,cv_data,cv_label,para_c,para_gamma):
    svc1=svm.SVC(kernel='poly',degree=3,coef0=20,C=para_gamma)   # create a svm with linear kernel, C=1, C is the parameter of penalty
    # svc1=svm.SVC(C=para_c,kernel='rbf',gamma=para_gamma)   # create a svm with linear kernel, C=1, C is the parameter of penalty
    svc1.fit(x,y.flatten())                                # train the svm, svm package will add the theta0 and x0
    preds=svc1.fit(x,y.flatten()).decision_function(cv_data)
    # preds=svc1.predict(cv_data)
    return svc1,preds,svc1.score(cv_data,cv_label.flatten())          # print the accurancy of svc1