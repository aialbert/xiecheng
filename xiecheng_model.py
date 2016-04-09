# -*- coding: utf-8 -*-
"""
Created on Tue Apr 05 14:40:43 2016

@author: albert
"""

from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import csv

#保存文件的写法 很重要
#np.savetxt("tt.csv",train_data,fmt="%s",delimiter = ',')


train_txt='./xiecheng_train_init.csv'
train_data=pd.read_csv(train_txt).values
m,n=train_data.shape 
train_x=train_data[:,1:n-1]
train_labels=train_data[:,-1]
feature_train,feature_test, target_train, target_test = train_test_split(train_x,train_labels, test_size=0.2, random_state=8)

tree_num=[50,100,500,1000,1500]
roc_auc_score=[]
tp=[]
fp=[]
tn=[]
fn=[]
recall=[]
precision=[]
for i in range(len(tree_num)):
    clf=RandomForestClassifier(n_estimators=tree_num[i],max_features=int(n/3),criterion='entropy',n_jobs=4)
    clf=clf.fit(feature_train, target_train)
    pre=clf.predict_proba(feature_test)
    pre_val=clf.predict(feature_test)       
    fpr,tpr,th=roc_curve(target_test,pre[:,1])
    roc_auc_score_t=auc(fpr,tpr)
    roc_auc_score.append(roc_auc_score_t)
    tp_t,fp_t,tn_t,fn_t,precision_t,recall_t=recall_precision(target_test,pre_val)
    tp.append(tp_t)
    fp.append(fp_t)
    tn.append(tn_t)
    recall.append(recall_t)
    precision.append(precision_t)
    

#pre_train=clf.predict_proba(feature_train)
#pre_val_train=clf.predict(feature_train)       
#fpr_train,tpr_train,th_train=roc_curve(target_train,pre_train[:,1])
#roc_auc_score_train=auc(fpr_train,tpr_train)
#tp_train,fp_train,tn_train,fn_train,precision_temp_train,recall_temp_train=recall_precision(target_train,pre_val_train)
