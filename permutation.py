import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from logistic import get_Logistic_paras
from logistic import sigmoid
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error

data=pd.read_excel('./data/表1_糖尿病_动脉硬化参数_edited4.xlsx')

data=np.array(data)
data=data[:,2:]                         # take off labels
np.random.shuffle(data)                 # shuffle the order of the array, evenly distribute, and each sample has an equal probability of being selected

labels=data[:,-1]

data=(data-data.mean(axis=0))/data.std(axis=0) #standardization
test_set=data[:20,:]                    # take the first 30 rows as the test set
cv_set=data[0:50,:]                    # take the first 30 rows as the
train_set=data[50:,:]                   # others the test set

test_labels=labels[:20]              
test_data=test_set[:,:-1]               

cv_labels=labels[0:50]              
cv_data=cv_set[:,:-1]                   # split training set labels and parameters

train_labels_logi=labels[50:]                # 200*14
train_data_logi=train_set[:,:-1]             # split training set labels and parameters

train_labels_ann=labels[50:]              
train_data_ann=train_set[:,:-1]             # split training set labels and parameters

train_labels_svm=labels[50:]              
train_data_svm=train_set[:,:-1]             # split training set labels and parameters

alpha=0.2
iters=5000
lamda=0

error_=np.zeros([1,14])

for k in range(10):
    data_copy=np.copy(train_data_logi)

    cost,theta=get_Logistic_paras(train_data_logi,train_labels_logi,alpha,iters,lamda)   # get the params of logistic-regretion model

    logistic_preds=sigmoid(theta,cv_data)
    logistic_preds_back=np.copy(logistic_preds)

    print(cv_labels,logistic_preds)
    FPR_log, TPR_log, thresholds_log = roc_curve(cv_labels,logistic_preds, pos_label=1)         # get roc curve
    area_log=AUC(cv_labels,logistic_preds)

    logistic_preds[logistic_preds>=0.5]=1
    logistic_preds[logistic_preds<0.5]=0
    acc=np.mean(logistic_preds==cv_labels)

    # print(cv_labels,'\n',logistic_preds,'\n','acc of logistic regresion is:',acc,' area=',area_log)
    print('acc of logistic regresion is:',acc,' area=',area_log,' precision=',precision_score(cv_labels,logistic_preds),
                                                            ' f1_score=',f1_score(cv_labels,logistic_preds),
                                                            ' recall=',recall_score(cv_labels,logistic_preds),
                                                            'rmse=',np.sqrt(mean_squared_error(cv_labels,logistic_preds)),
                                                            '\n*********************************************************')
    note_acc=np.copy(acc)
    note=np.sqrt(mean_squared_error(cv_labels,logistic_preds))

    maze_copy=np.copy(data_copy)
    np.random.shuffle(maze_copy)

    for i in range(14):
        print(i+1)
        loop_copy=np.copy(data_copy)
        loop_copy[:,i]=maze_copy[:,i]
        cost,theta=get_Logistic_paras(loop_copy,train_labels_logi,alpha,iters,lamda)   # get the params of logistic-regretion model

        logistic_preds=sigmoid(theta,cv_data)
        logistic_preds_back=np.copy(logistic_preds)

        FPR_log, TPR_log, thresholds_log = roc_curve(cv_labels,logistic_preds, pos_label=1)         # get roc curve
        area_log=AUC(cv_labels,logistic_preds)

        logistic_preds[logistic_preds>=0.5]=1
        logistic_preds[logistic_preds<0.5]=0
        acc=np.mean(logistic_preds==cv_labels)

        if note_acc-acc>0:
            error_[0,i]+=1
        # print(cv_labels,'\n',logistic_preds,'\n','acc of logistic regresion is:',acc,' area=',area_log)
        print('acc of logistic regresion is:',acc,' area=',area_log,' precision=',precision_score(cv_labels,logistic_preds),
                                                                ' f1_score=',f1_score(cv_labels,logistic_preds),
                                                                ' recall=',recall_score(cv_labels,logistic_preds),
                                                                'D_rmse=',note-np.sqrt(mean_squared_error(cv_labels,logistic_preds)),
                                                                '\n*********************************************************')

print(error_)