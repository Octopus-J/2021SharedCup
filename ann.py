import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn

def get_k_fold_data(k,i,x,y):     # return the train and test data needed in the kth fold cross-validation
    assert k>1
    fold_size = x.shape[0]//k     # integer division
    x_train,y_train=None,None 
    for j in range(k):            # divide data into k parts and combine them into train set and test set
        indx=slice(j*fold_size,(j+1)*fold_size) 
        x_part=x[indx,:]
        y_part=y[indx]            # slice data, get the kth part data
        if i==j:
            x_valid=x_part
            y_valid=y_part
        elif x_train is None:
            x_train,y_train=x_part,y_part
        else:
            x_train=torch.cat((x_train,x_part),dim=0)
            y_train=torch.cat((y_train,y_part),dim=0)
    return x_train,y_train,x_valid,y_valid

def k_fold(k,x_train,y_train,num_epochs,learning_rate,weight_decay,batch_size):    # define k-fold cross validation function
    train_l_sum,valid_l_sum=0,0                                                    # container to save log rmse
    for i in range(k):
        net=get_net(x_train.shape[1])
        data=get_k_fold_data(k,i,x_train,y_train)
        train_ls,test_ls=train(net,*data,num_epochs,learning_rate,weight_decay,batch_size)
        train_l_sum+=train_ls[-1]
        valid_l_sum+=test_ls[-1]                                                   # record the kth fold log rmse
        #print('fold %d, train rmase %f, valid rmse %f' % (i,train_ls[-1],test_ls[-1]))
    return train_l_sum/k,valid_l_sum/k 

def get_net(feature_num):                           # use the simple linear regression model
    net=nn.Sequential(nn.Linear(feature_num,32),nn.BatchNorm1d(32),nn.ReLU(),
                        nn.Linear(32,64),nn.BatchNorm1d(64),nn.ReLU(),
                        nn.Linear(64,32),nn.BatchNorm1d(32),nn.ReLU(),
                        nn.Linear(32,1))
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=5)    # initialize the weight parameter by normal disttribution
    return net

def log_rmse(net,features,label):                                # use log to get ratative error
    loss=nn.MSELoss()                                            # difine loss function
    clipped_predicts=torch.clamp(net(features),1,float('inf'))   # turn the predicts which less than 1 to 1
    rmse=torch.sqrt(loss(torch.log(clipped_predicts),torch.log(label)))
    return rmse

def train(net,train_features,train_labels,test_features,test_labels,num_epochs,learning_rate,weight_decay,batch_size):
    loss=nn.MSELoss()                                            # difine loss function
    train_ls,test_ls=[],[]                                                     # record the error of train set and test set
    optimizer = torch.optim.Adam(params=net.parameters(),lr=learning_rate,weight_decay=weight_decay)   # use Adam optimizer
    dataset = torch.utils.data.TensorDataset(train_features,train_labels)      # create a dataset
    train_iters = torch.utils.data.DataLoader(dataset,batch_size,shuffle=True) # load dataset 
    net = net.float()      

    for epoch in range(num_epochs):                      # tain num_epochs times          
        for x,y in train_iters:
            optimizer.zero_grad()                        # initialize grad to 0 each epoch
            x=x*1000
            inp=net(x.float())
            l=loss(inp.reshape(-1),y.float())            # caculate loss
            if epoch==0 or epoch==1:
                for name, parameters in net.named_parameters():
                    print(name, ':', parameters)
                print('x=',x[0:2,:],'\nnet(x)=',inp.T,'\nloss=',l)
            l.backward()                                 # backward
            optimizer.step()                             # next batch
        train_ls.append(l)  #record the log rmse after each epoch
        if test_labels is not None:
            test_ls.append(l)

    return train_ls,test_ls

def predict(train_features,train_labels,test_features,data_test,num_epochs,learning_rate,weight_decay,batch_size):
    train_features = torch.tensor(train_features,dtype=torch.float32)
    train_labels = torch.tensor(train_labels,dtype=torch.float32)
    test_features = torch.tensor(test_features,dtype=torch.float32)
    net=get_net(train_features.shape[1])
    train_ls,_=train(net,train_features,train_labels,None,None,num_epochs,learning_rate,weight_decay,batch_size)
    preds=net(test_features).detach().numpy()

    return net,train_ls,preds

