import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso


def get_lasso_paras(train_data,train_label,alpha,iters,lamda):
    note_lmda=np.zeros([2000,train_data.shape[1]])
    i=0
    for lmda in np.arange(0.,1,0.0005):
        lasso=Lasso(alpha=lmda)
        lasso.fit(train_data,train_label)
        theta=lasso.coef_
        #cost,theta=gradientDescent(theta,train_data,train_label,alpha,iters,lamda=lmda)
        note_lmda[i,:]=theta
        i+=1

    return note_lmda

kk=0
if kk==1:
    data=pd.read_excel('./data/表1_糖尿病_动脉硬化参数_edited4.xlsx')
    data=data[['','','','','','','A_S']]

    data=np.array(data)
    data=data[:,1:]                         # take off labels
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

    train_labels_logi=labels[50:100]              
    train_data_logi=train_set[:50,:-1]             # split training set labels and parameters

    train_labels_ann=labels[100:150]              
    train_data_ann=train_set[50:100,:-1]             # split training set labels and parameters

    train_labels_svm=labels[150:]              
    train_data_svm=train_set[100:,:-1]             # split training set labels and parameters

    alpha=0.2
    iters=5000
    lamda=0.1
    #########################################################################################################################################
    para_names=['AGE','BP_HIGH','HYPERTENTION','GLU','HBA1C','TG','TC','HDL_C','LDL_C','LDL/HDL','FBG','CRP','BMI','CP']
    all_theta=get_lasso_paras(data[:,:-1],labels,alpha,iters,lamda)

    fig,ax = plt.subplots()			#create a draw instance 
    for i in range(14):
        ax.plot(np.arange(0.,1,0.0005),all_theta[:,i],label=para_names[i])
    ax.set(ylabel='Coefficients',xlabel='Lamda',title='Coefficients-Lamda curve')
    plt.legend()
    plt.xlim(0,0.2)
    plt.show()