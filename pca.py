from itertools import combinations
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data=pd.read_excel('./data/表1_糖尿病_动脉硬化参数_edited4.xlsx')

np.set_printoptions(threshold=np.inf)

data=data[['BP_HIGH','CP','LDL_C','TG','CRP','AGE','GLU','LDL/HDL','A_S']]
data=data[['FBG','AGE','CP','HYPERTENTION','LDL/HDL','TC','CRP','A_S']]

data=data[['HYPERTENTION','BP_HIGH','AGE','FBG','TG','A_S']]
data=data[['BMI','HBA1C','AGE','FBG','BP_HIGH','TG','A_S']]
data=np.array(data)
data=data[:,2:]                         # take off labels

np.random.shuffle(data)                 # shuffle the order of the array, evenly distribute, and each sample has an equal probability of being selected
labels=data[:,-1]

# data=(data-np.min(data,axis=0))/(np.max(data,axis=0)-np.min(data,axis=0)) #normalization
# data= np.array( [[row[i] for i in range(0, 15) if i != col] for row in data] )
data=(data-data.mean(axis=0))/data.std(axis=0) #standardization
data2=data[:,:-1]


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

print(data2.shape,'\n',data2[0,:])
data2=data2.T
pca=PCA(n_components=0.95)
pca.fit(data2)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_)
print(pca.components_.T.shape)
print(pca.components_.T[0,:])
