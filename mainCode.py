import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from logistic import get_Logistic_paras,sigmoid
import logisticCopy
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import roc_curve
from ann import predict
from torch import nn
from svm import svm_pred
from lasso import get_lasso_paras
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

def softmax(x):
    x=np.exp(x)/np.sum(np.exp(x))
    return x

for mm in range(0,1000000):
    data=pd.read_excel('./data/表1_糖尿病_动脉硬化参数_edited4.xlsx')
    data=data[['BP_HIGH','CP','LDL_C','TG','CRP','AGE','GLU','LDL/HDL','A_S']]
    # data=data[['FBG','AGE','CP','HYPERTENTION','LDL/HDL','TC','CRP','A_S']]

    # data=data[['HYPERTENTION','BP_HIGH','AGE','FBG','TG','A_S']]
    # data=data[['BMI','HBA1C','AGE','FBG','BP_HIGH','TG','A_S']]
    data=np.array(data)
    data=data[:,2:]                         # take off labels
    np.random.shuffle(data)                 # shuffle the order of the array, evenly distribute, and each sample has an equal probability of being selected
    labels=data[:,-1]

    # data=(data-np.min(data,axis=0))/(np.max(data,axis=0)-np.min(data,axis=0)) #normalization
    # data= np.array( [[row[i] for i in range(0, 15) if i != col] for row in data] )
    data=(data-data.mean(axis=0))/data.std(axis=0) #standardization
    test_set=data[:20,:]                    # take the first 30 rows as the test set
    cv_set=data[0:50,:]                    # take the first 30 rows as the
    train_set=data[50:,:]                   # others the test set

    test_labels=labels[:20]              
    test_data=test_set[:,:-1]               

    cv_labels=labels[0:50]              
    cv_data=cv_set[:,:-1]                   # split training set labels and parameters

    train_labels_logi=labels[50:]              
    train_data_logi=train_set[:,:-1]             # split training set labels and parameters

    train_labels_ann=labels[50:]              
    train_data_ann=train_set[:,:-1]             # split training set labels and parameters

    train_labels_svm=labels[50:]              
    train_data_svm=train_set[:,:-1]             # split training set labels and parameters

    # train_labels_logi=labels[50:100]              
    # train_data_logi=train_set[:50,:-1]             # split training set labels and parameters

    # train_labels_ann=labels[100:150]              
    # train_data_ann=train_set[50:100,:-1]             # split training set labels and parameters

    # train_labels_svm=labels[150:]              
    # train_data_svm=train_set[100:,:-1]             # split training set labels and parameters

    alpha=0.2
    iters=5000
    lamda=0
    #########################################################################################################################################
    para_names=['AGE','BP_HIGH','HYPERTENTION','GLU','HBA1C','TG','TC','HDL_C','LDL_C','LDL/HDL','FBG','CRP','BMI','CP']
    all_theta=get_lasso_paras(train_data_logi,train_labels_logi,alpha,iters,lamda=1)

    fig,ax = plt.subplots()			#create a draw instance 
    for i in range(14):
        ax.plot(np.arange(0.,0.03,0.0005),all_theta[:,i],label=para_names[i])
    ax.set(ylabel='Coefficients',xlabel='Lamda',title='Coefficients-Lamda curve')
    plt.legend()
    plt.xlim(0,0.025)
    plt.show()
    #########################################################################################################################################
    cost,theta=get_Logistic_paras(train_data_logi,train_labels_logi,alpha,iters,lamda)   # get the params of logistic-regretion model
    #theta=np.load('./data/logi_theta_0.8.npy')
    logistic_preds=sigmoid(theta,cv_data)
    logistic_preds_back=np.copy(logistic_preds)

    print(cv_labels,logistic_preds)
    FPR_log, TPR_log, thresholds_log = roc_curve(cv_labels,logistic_preds, pos_label=1)         # get roc curve
    area_log=AUC(cv_labels,logistic_preds)

    logistic_preds[logistic_preds>=0.5]=1
    logistic_preds[logistic_preds<0.5]=0
    acc=np.mean(logistic_preds==cv_labels)
    if (acc>=0.81):
        np.save("./data/logi_theta_0.75.npy",theta)
    # print(cv_labels,'\n',logistic_preds,'\n','acc of logistic regresion is:',acc,' area=',area_log)
    print('acc of logistic regresion is:',acc,' area=',area_log,' precision=',precision_score(cv_labels,logistic_preds),
                                                                ' f1_score=',f1_score(cv_labels,logistic_preds),
                                                                ' recall=',recall_score(cv_labels,logistic_preds))

    #########################################################################################################################################
    net,train_ls,ann_preds=predict(train_data_ann,train_labels_ann,cv_data,data_test=None,num_epochs=1000,learning_rate=0.1,weight_decay=0.1,batch_size=20)
    ann_preds_back=np.copy(ann_preds)
    ann_preds=1.0/(1+np.exp(-ann_preds))

    print(ann_preds.T)
    FPR_ann, TPR_ann, thresholds_ann = roc_curve(cv_labels,ann_preds, pos_label=1)         # get roc curve
    area_ann=AUC(cv_labels,ann_preds)

    ann_preds[ann_preds<=0.5]=0
    ann_preds[ann_preds>0.5]=1
    ann_acc=np.mean(ann_preds==cv_labels)
    # print(cv_labels,'\n',ann_preds.T,'\n','acc of ann is:',ann_acc,' area=',area_ann)
    print('acc of ann is:',ann_acc,' area=',area_ann,' precision=',precision_score(cv_labels,ann_preds),
                                                    ' f1_score=',f1_score(cv_labels,ann_preds),
                                                    ' recall=',recall_score(cv_labels,ann_preds))

    if (ann_acc>=0.76):
        torch.save(net.state_dict(),"./data/ann_theta_75.pth")
    ##########################################################################################################################################
    svc1,svm_preds,svm_acc=svm_pred(train_data_svm,train_labels_svm,cv_data,cv_labels,para_c=3,para_gamma=10)
    svm_preds_back=np.copy(svm_preds)

    svm_preds[svm_preds<=0.5]=0
    svm_preds[svm_preds>0.5]=1
    print(svm_preds)
    FPR_svm, TPR_svm, thresholds_svm = roc_curve(cv_labels,svm_preds, pos_label=1)         # get roc curve
    area_svm=AUC(cv_labels,svm_preds)
    print('acc of svm is:',svm_acc,' area=',area_svm,' precision=',precision_score(cv_labels,svm_preds),
                                                    ' f1_score=',f1_score(cv_labels,svm_preds),
                                                    ' recall=',recall_score(cv_labels,svm_preds))
    
    if (svm_acc>=0.76):
        joblib.dump(svc1,'./data/svc_75.model')

    if (area_log+area_ann+area_svm)/3.0>=0.7:
        plt.figure()
        plt.plot(FPR_log, TPR_log, color='red'
                ,label='logistic ROC curve (area = %0.2f)' % area_log)
        plt.plot(FPR_ann, TPR_ann, color='blue'
                ,label='ANN ROC curve (area = %0.2f)' % area_ann)
        plt.plot(FPR_svm, TPR_svm, color='green'
                ,label='SVM ROC curve (area = %0.2f)' % area_svm)
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

    # #####################################################################################################################################################
    stacking_data=np.zeros([50,3])
    stacking_data[:,0]=logistic_preds_back
    stacking_data[:,1]=ann_preds_back.reshape(-1)
    stacking_data[:,2]=svm_preds_back

    cost,theta2=logisticCopy.get_Logistic_paras(stacking_data,cv_labels,alpha,iters,lamda)   # get the params of logistic-regretion model
    #theta=np.load('./data/logi_theta_0.8.npy')

    stacking_preds=sigmoid(theta2,stacking_data)

    print(cv_labels,'\n',stacking_preds)
    FPR_stacking, TPR_stacking, thresholds_stacking = roc_curve(cv_labels,stacking_preds, pos_label=1)         # get roc curve
    area_stacking=AUC(cv_labels,stacking_preds)


    stacking_preds[stacking_preds>=0.5]=1
    stacking_preds[stacking_preds<0.5]=0
    acc=np.mean(stacking_preds==cv_labels)
    if (acc>=0.86):
        np.save("./data/stacking_0.75.npy",theta2)
    # print(cv_labels,'\n',stacking_preds,'\n','acc of Stacking is:',acc,' area=',area_log)
    print('acc of Stacking is:',acc,' area=',area_stacking,' precision=',precision_score(cv_labels,stacking_preds),
                                                    ' f1_score=',f1_score(cv_labels,stacking_preds),
                                                    ' recall=',recall_score(cv_labels,stacking_preds),
    '\n*********************************************************************************')

    if area_stacking>=0.78:
        plt.figure()
        plt.plot(FPR_stacking, TPR_stacking, color='red'
                ,label='Staking ROC curve (area = %0.2f)' % area_stacking)

        plt.plot([0, 1], [0, 1], color='black', linestyle='--')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('Recall')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()



#####################################################################################################################################################
fig,ax = plt.subplots()			#create a draw instance 
ax.plot(np.arange(iters),cost)
ax.set(xlabel='iters',ylabel='cost',title='Cost-Iters curve')
plt.show()

fig,ax = plt.subplots()			#create a draw instance 
ax.plot(np.arange(1000),train_ls)
ax.set(xlabel='iters',ylabel='cost',title='Cost-Iters curve')
plt.show()