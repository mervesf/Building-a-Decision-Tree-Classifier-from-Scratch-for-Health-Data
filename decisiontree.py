# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 11:17:25 2023

@author: user
"""

import pandas as pd
import numpy as np
import math
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


df=pd.read_csv('data (2).csv')
df.drop(["id","Unnamed: 32"],axis=1,inplace=True)

df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]
y = df[['diagnosis']]
x = df.drop(["diagnosis"],axis=1)
x=(x-np.min(x))/(np.max(x)-np.min(x))


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)


def calculate_gini(x_data,x_train,y_train):
    class_first=x_data.loc[y_train[y_train.iloc[:,0]==y_train.value_counts().index[0]].index]
    class_second=x_data.loc[y_train[y_train.iloc[:,0]==y_train.value_counts().index[1]].index]
    unique=x_data[0].value_counts().index
    gini_list=[]
    for i in range(0,len(unique)):
        first_class_rate=np.square((len(class_first[class_first.iloc[:,0]==unique[i]]))/(len(x_data[x_data.iloc[:,0]==unique[i]])))
        second_class_rate=np.square((len(class_second[class_second.iloc[:,0]==unique[i]]))/(len(x_data[x_data.iloc[:,0]==unique[i]])))
        
        gini_list.append((1-first_class_rate-second_class_rate)*(len(x_data[x_data==unique[i]])/len(x_data)))
    return sum(gini_list)

def find_min_index(value,indexs):
    new_list=sorted(value)
    for i in range(0,len(value)):
         if value[i]==new_list[0]:   
           return indexs[i]
def find_max_index(value,indexs):
    new_list=sorted(value)
    for i in range(0,len(value)):
         if value[i]==new_list[-1]:   
           return indexs[i]
def calculate_entropy(x_data,y_data,threshold):
    new_list=['Upper' if x_data[i]>=threshold else 'Lower' for i in x_data.index]
    new_data=pd.DataFrame(new_list,index=x_data.index)
    new_data=pd.concat([new_data, y_data],axis=1)
    unique=new_data.iloc[:,0].unique()
    result=[]
    epsilon = 1e-10
    for i in unique:
        value=len(new_data[(new_data.iloc[:,1]==new_data.iloc[:,1].value_counts().index[0])&(new_data.iloc[:,0]==i)])
        value/=len(new_data[new_data.iloc[:,0]==i])
        value_1=len(new_data[(new_data.iloc[:,1]==new_data.iloc[:,1].value_counts().index[1])&(new_data.iloc[:,0]==i)])
        value_1/=len(new_data[new_data.iloc[:,0]==i])
        result.append((len(new_data[new_data.iloc[:,0]==i])/len(new_data))*((-value*np.log2(value+epsilon))+(-value_1*np.log2(value_1+epsilon))))
    return sum(result)
        
def calculate_information_gain(x_train,y_train,opti_threshold_features):
    first_class=(len(y_train[y_train.iloc[:,0]==y_train.value_counts().index[0]])/len(y_train))*(np.log10(len(y_train[y_train.iloc[:,0]==y_train.value_counts().index[0]])/len(y_train)))
    second_class=(len(y_train[y_train.iloc[:,0]==y_train.value_counts().index[1]])/len(y_train))*(np.log10(len(y_train[y_train.iloc[:,0]==y_train.value_counts().index[1]])/len(y_train)))
    entropy=-first_class-second_class
    information_gain=[]
    
    for i in range(0,len(x_train.columns)):
        
        
        entropy=entropy-calculate_entropy(x_train.iloc[:,i],y_train,opti_threshold_features[i])
        information_gain.append(entropy)
    root=find_max_index(information_gain,list(range(x_train.shape[1])))
    return root
def split_numeric_feature(x_train,y_train,indis):
    gini_list_threshold=[]
    gini_threshold=[]
    x_data=pd.DataFrame(np.full(len(x_train), 1),index=x_train.index)
    mean=np.mean(x_train.iloc[:,indis])
    std=math.ceil((np.std(x_train.iloc[:,indis])))
    if mean>=2:
        for i in range(std,math.floor(np.max(x_train.iloc[:,indis])),std):
            if (len(x_train.loc[x_train.iloc[:,indis]>i])<10) or (len(x_train.loc[x_train.iloc[:,indis]<=i])<10):
                
                continue
            else:
                x_data[0]=['Upper' if j>=i else 'Lower' for j in x_train.iloc[:,indis]]
                value_gini=calculate_gini(x_data,x_train,y_train)
                gini_list_threshold.append(value_gini)
                gini_threshold.append(i)
    else:
        for i in np.arange(np.min(x_train.iloc[:,indis]),math.floor(np.max(x_train.iloc[:,indis])),0.07):
            if (len(x_train.loc[x_train.iloc[:,indis]>i])<10) or (len(x_train.loc[x_train.iloc[:,indis]<=i])<10):
                continue
            else:
                x_data[0]=['Upper' if j>=i else 'Lower' for j in x_train.iloc[:,indis]]
                value_gini=calculate_gini(x_data,x_train,y_train)
                gini_list_threshold.append(value_gini)
                gini_threshold.append(i)

    result=find_min_index(gini_list_threshold,gini_threshold)
    return result 
def feature_entropy(index,y_data):

    class_first=len(y_data[y_data.loc[index,:]==y_data.loc[index,:].value_counts().index[0]].dropna(axis=0))
    class_second=len(y_data[y_data.loc[index,:]==y_data.loc[index,:].value_counts().index[1]].dropna(axis=0))
    result_entro=-((class_first/len(index))*np.log2(class_first/len(index)))-((class_second/len(index))*np.log2(class_second/len(index)))
    return result_entro

def decision_(x_train,y_train):
    opti_threshold_features=[]
    result_1=[]
    for i in range(0,len(x_train.columns)):
        result1=split_numeric_feature(x_train,y_train,i)
        if pd.isna(result1):
           opti_threshold_features.append(np.mean(x_train.iloc[:,i]))
        else:
           opti_threshold_features.append(result1)
    calculate_entropy_1=[]
    root_feature=calculate_information_gain(x_train,y_train,opti_threshold_features)
    calculate_entropy_1.append(x_train.loc[x_train.iloc[:,root_feature]>opti_threshold_features[root_feature]].index)
    calculate_entropy_1.append(x_train.loc[x_train.iloc[:,root_feature]<=opti_threshold_features[root_feature]].index)

    for i in range(0,len(calculate_entropy_1)):
        result_1.append(feature_entropy(calculate_entropy_1[i],y_train))
    
    return result_1,calculate_entropy_1,root_feature,opti_threshold_features[root_feature]
def recursif(back_list,x_train,point,depth_1,residual,class_list,max_depth,y_train):
    is_class=False
    is_final=True
    result_1,split_index,root_feature,threshold=decision_(x_train.loc[back_list[-1][0],:],y_train.loc[back_list[-1][0],:])
    back_list.pop(-1)
    for i in range(0,len(result_1)):
        if result_1[i]<0.2:
            point.append([depth_1+1,root_feature,threshold,i,True])
            class_list.append(point.copy())
            point.pop(-1)
            is_class=True
        else:
            back_list.append([split_index[i],i,root_feature,threshold])
            is_final=False
    if not is_class:
        li_1=[0,1]
        li_1.remove(back_list[-1][1])
        residual.append([depth_1+1,back_list[-1][2],back_list[-1][3],li_1[0],is_class])
    
    
    if is_final:
        new_point=[]
        for i in range(0,len(point)):
            if point[i][0]==residual[-1][0]:
                new_point.append(residual[-1])
                residual.pop(-1)
                break
            else:                                                                           
                new_point.append(point[i])
        point=new_point
        depth=point[-1][0]
    else:
        point.append([depth_1+1,back_list[-1][2],back_list[-1][3],back_list[-1][1],is_class])
    depth_1+=1
    if depth_1>max_depth:
        new_list=[]
        new_list_2=point
        for i in range(0,len(residual)):
                for j in range(0,len(new_list_2)):
                    if (residual[i][0]==new_list_2[j][0])&(j==0):
                        new_list.append(residual[i])
                        break
                    elif(residual[i][0]==new_list_2[j][0]):
                        new_list_2.pop(-1)
                        new_list_2.append(residual[i])
                        new_list.append([new_list_2[:j+1]])
                        break
        return point,class_list,new_list
    return recursif(back_list,x_train,point,depth_1,residual,class_list,max_depth,y_train)

def decision_tree(x_train,y_train,x_test,y_test,depth):
        y_head=pd.DataFrame(np.full(x_test.shape[0],9),index=y_test.index)
        number_depth=1
        class_list=[]
        point=[]
        back_list=[]
        residual=[]
        back_list.append([x_train.index])
        point_1,class_list_1,new_list_1=recursif(back_list,x_train,point,number_depth,residual,class_list,depth,y_train)
        x_final=x_test
        for i in range(0,len(point_1)):
            if point_1[i][3]==0:
                x_final=x_final.loc[x_final.iloc[:,point_1[i][1]]>point_1[i][2]]
            else:
                x_final=x_final.loc[x_final.iloc[:,point_1[i][1]]<=point_1[i][2]]
        y_head.loc[x_final.index,:]=y_test.loc[x_final.index,:].value_counts().index[0][0]
    
        for i in range(0,len(new_list_1[0][0])):
            if new_list_1[0][0][i][3]==0:
                x_final=x_final.loc[x_final.iloc[:,new_list_1[0][0][i][1]]>new_list_1[0][0][i][2]]
            else:
                x_final=x_final.loc[x_final.iloc[:,new_list_1[0][0][i][1]]<=new_list_1[0][0][i][2]]
            y_head.loc[x_final.index,:]=y_test.loc[x_final.index,:].value_counts().index[0][0]
            x_final=x_test
        x_final=x_test
        for i in range(0,len(class_list_1)):
            if class_list_1[i][i][3]==0:
                x_final=x_final.loc[x_final.iloc[:,class_list_1[i][i][1]]>class_list_1[i][i][2]]
            else:
                x_final=x_final.loc[x_final.iloc[:,class_list_1[i][i][1]]<=class_list_1[i][i][2]]
            y_head.loc[x_final.index,:]=y_test.loc[x_final.index,:].value_counts().index[0][0]
            x_final=x_test
            
        return y_head
    

            
sonuc_decision=decision_tree(x_train,y_train,x_test,y_test,4)           
print(accuracy_score(y_test, sonuc_decision))          


    
    