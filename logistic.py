import numpy as np

def sigmoid(theta,x):           # the hypothesis of logistic regression
    z=x@theta
    z[z>=650]=650
    z[z<-650]=-650
    return 1.0/(1+np.exp(-z))     # the sigmoid function

def costFunction(theta,x,y,lamda=0,flag=0,):    # flag is used to control process,and lamda, the regularization parameter
    hyp=sigmoid(theta,x)
    epsilon=1.0e-200                           # epsilon is a very small constan, the reason to use it is to avoid the np.log(0),this problem troubled me a lot
    if flag==0:
        cost1=-(y.T)*(np.log(hyp+epsilon))
        cost2=-(1-y)*(np.log(1-hyp+epsilon))
        cost=np.sum(cost1+cost2)/len(y)
    else:
        _theta=theta[1:]                       
        _theta=np.insert(_theta,0,0)           # in python, if a parameter is variable,then in the transfer behavior,itself could be changed
        cost1=-(y.T)*(np.log(hyp+epsilon))
        cost2=-(1-y)*(np.log(1-hyp+epsilon))   
        cost=np.sum(cost1+cost2)/(2*len(y))+(lamda/(2*len(y)))*np.sum(np.power(_theta,2))  # L2

    return cost

def gradientDescent(theta,x,y,alpha,iters,lamda=0,flag=0,):   
    m=len(y)
    cost=np.zeros(iters)
    if (flag==0):       # no regularization
        for i in range(0,iters):
            sig=sigmoid(theta,x)
            theta=theta-(alpha/m)*(x.T@(sig-y))
            cost[i]=costFunction(theta,x,y)      # record the cost value in each iteration
    else:               # with regularization
        for i in range(0,iters):
            _theta=theta[1:]                                  # we didn't punish the theta0, so in the graident descent process, it's 0
            _theta=np.insert(_theta,0,0)                      # notice!,here we can't use _theta[0]=0,because in python if a=1,b=a, then both a and b is the pointer of 1
            sig=sigmoid(theta,x)
            para1=(alpha/m)*(x.T@(sig-y))
            para2=(alpha*lamda/m)*_theta
            theta=theta-para1-para2
            cost[i]=costFunction(theta,x,y,flag,lamda)      # record the cost value in each iteration

    return cost,theta 

def get_Logistic_paras(train_data,train_label,alpha,iters,lamda):
    [num_samples,num_features]=train_data.shape
    theta=np.random.randint(-10,10,num_features)
    cost,theta=gradientDescent(theta,train_data,train_label,alpha,iters,lamda,flag=1)
    return cost,theta