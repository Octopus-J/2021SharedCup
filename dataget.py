import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

filepath='./data/表1_糖尿病_动脉硬化参数_edited2.xlsx'

def dataget(filepath):
        data=pd.read_excel(filepath)
        #print(data)
        return data

data=dataget(filepath)
meanVal=np.nanmean(data,axis=0)      # mean value which don't include nan
stdVal=np.nanstd(data,axis=0)        # std value
data=(data-meanVal)/stdVal           # z-score standardization

minVal=np.nanmin(data,axis=0) #
maxVal=np.nanmax(data,axis=0) #
data=(data-minVal)/(maxVal-minVal)           # normorlization

n_neighbor=3
imputer=KNNImputer(n_neighbors=n_neighbor)
data[['CRP','AGE','BP_HIGH','HYPERTENTION','A_S','GLU','HBA1C',              # fix up the CRP
        'TG','TC','HDL_C','LDL_C','LDL/HDL','FBG']]=imputer.fit_transform(data[['CRP','AGE','BP_HIGH','HYPERTENTION','A_S','GLU','HBA1C',
                                                                                        'TG','TC','HDL_C','LDL_C','LDL/HDL','FBG']])

data[['BMI','AGE','BP_HIGH','HYPERTENTION','A_S','GLU','HBA1C',              # fix up the BMI
        'TG','TC','HDL_C','LDL_C','LDL/HDL','FBG']]=imputer.fit_transform(data[['BMI','AGE','BP_HIGH','HYPERTENTION','A_S','GLU','HBA1C',
                                                                                        'TG','TC','HDL_C','LDL_C','LDL/HDL','FBG']])

data[['CP','AGE','BP_HIGH','HYPERTENTION','A_S','GLU','HBA1C',              # fix up the CP
        'TG','TC','HDL_C','LDL_C','LDL/HDL','FBG']]=imputer.fit_transform(data[['CP','AGE','BP_HIGH','HYPERTENTION','A_S','GLU','HBA1C',
                                                                                        'TG','TC','HDL_C','LDL_C','LDL/HDL','FBG']])

data[['TG','AGE','BP_HIGH','HYPERTENTION','A_S','GLU','HBA1C',               # fix up the TG
        'TC','HDL_C','LDL_C','LDL/HDL','FBG']]=imputer.fit_transform(data[['TG','AGE','BP_HIGH','HYPERTENTION','A_S','GLU','HBA1C',
                                                                                        'TC','HDL_C','LDL_C','LDL/HDL','FBG']])

data[['TC','AGE','BP_HIGH','HYPERTENTION','A_S','GLU','HBA1C',               # fix up the CTC
        'TG','HDL_C','LDL_C','LDL/HDL','FBG']]=imputer.fit_transform(data[['TC','AGE','BP_HIGH','HYPERTENTION','A_S','GLU','HBA1C',
                                                                                        'TG','HDL_C','LDL_C','LDL/HDL','FBG']])

data[['HDL_C','AGE','BP_HIGH','HYPERTENTION','A_S','GLU','HBA1C',            # fix up the HDL_C
        'TG','TC','LDL_C','LDL/HDL','FBG']]=imputer.fit_transform(data[['HDL_C','AGE','BP_HIGH','HYPERTENTION','A_S','GLU','HBA1C',
                                                                                        'TG','TC','LDL_C','LDL/HDL','FBG']])

data[['LDL_C','AGE','BP_HIGH','HYPERTENTION','A_S','GLU','HBA1C',            # fix up the LDL_C
        'TG','TC','HDL_C','LDL/HDL','FBG']]=imputer.fit_transform(data[['LDL_C','AGE','BP_HIGH','HYPERTENTION','A_S','GLU','HBA1C',
                                                                                        'TG','TC','HDL_C','LDL/HDL','FBG']])

data[['FBG','AGE','BP_HIGH','HYPERTENTION','A_S','GLU','HBA1C',              # fix up the FBG
        'TG','TC','HDL_C','LDL/HDL']]=imputer.fit_transform(data[['FBG','AGE','BP_HIGH','HYPERTENTION','A_S','GLU','HBA1C',
                                                                                        'TG','TC','HDL_C','LDL/HDL']])

data[['SCR','AGE','BP_HIGH','HYPERTENTION','A_S','GLU','HBA1C',              # fix up the FBG
        'TG','TC','HDL_C','LDL/HDL']]=imputer.fit_transform(data[['SCR','AGE','BP_HIGH','HYPERTENTION','A_S','GLU','HBA1C',
                                                                                        'TG','TC','HDL_C','LDL/HDL']])


data=(data*(maxVal-minVal))+minVal           # denormalization
data=(data*stdVal)+meanVal           # destandardization

tempVal=np.array([0.321,0.325,2.15,1.1,0.33,0.1,0.735,0.2,0.538,0.4])
tempIdx=np.array([2,5,8,9,10,11,12,15,16,20])
note=0
for j in range(0,10):
        val=tempVal[j]
        idx=tempIdx[j]
        note+=abs(data.at[idx,'CRP']-val)

print('note(',n_neighbor,')=',note)
#print('note(',n_neighbor,')=',note,'\n',data.iloc[7])
data.to_excel('./data/表1_糖尿病_动脉硬化参数_edited3.xlsx')

# get params by loop
databack=pd.copy(data)
note=np.zeros(19)
name=['temp2','temp3','temp4','temp5','temp6','temp7','temp8','temp9','temp10','temp11','temp12','temp13','temp14','temp15','temp16','temp17','temp18','temp19','temp20']
for i in range(2,21):
        temName=name[i-2]
        imputer=KNNImputer(n_neighbors=i)
        data[[temName,'AGE','BP_HIGH','HYPERTENTION','A_S','GLU','HBA1C',              # 用其余参数对CRP进行补全
        'TG','TC','HDL_C','LDL_C','LDL/HDL','FBG','TP']]=imputer.fit_transform(data[[temName,'AGE','BP_HIGH','HYPERTENTION','A_S','GLU','HBA1C',
                                                                                        'TG','TC','HDL_C','LDL_C','LDL/HDL','FBG','TP']])
for j in range(0,10):
        val=tempValue[j]
        idx=tempIndex[j]
        print(idx,val,data.at[idx,temName])
        note[i-2]+=(data.at[idx,temName]-val)
print(note)