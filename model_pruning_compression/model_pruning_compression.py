#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# EVSense: Xudong Wang, Guoming Tang
# time:2021/5/29

# In[ ]:
import os
import re 
import numpy as np
import time
import timeit
import datetime
import pandas as pd
import pickle 
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune
torch.set_default_tensor_type('torch.FloatTensor')

import sys 
sys.path.append("..") 

##Self Implement package
from ..data_processing import TimeseriesDataset, get_resident_dt, data_scalar, train_test_data_split 
from ..utils import * 
from ..session_analysis import *
from ..model.models import EVsense_dummy_network,EVsense_DNN
from ..experiment import experiment, experiment_lr
from ..model.metrics import *   
from ..model.loss import * 

## Set pandas dataFrame to show all the columns 
pd.set_option('display.max_columns', None)

## Fix random state
setup_seed(0)  


# In[ ]:

with open('../pickle_data/661.pkl', 'rb') as f:
    dt_661_1min = pickle.load(f) 


# In[ ]:


model = torch.load('../global-state.pth',map_location=torch.device('cpu'))
print(model)


# In[ ]:

# Select time period

time_split_661 = {'train':{'start':'2018-05-01','end':'2018-06-30'},
              'test':{'start':'2018-07-01','end':'2018-07-31'}}
agg_p_train_661, label_train_661, agg_p_test_661, label_test_661 = train_test_data_split(dt_661_1min,time_split_661) 


 ## Load data for Pytorch
train_dataset_661 = TimeseriesDataset(agg_p_train_661,label_train_661,seq_len= 20)
train_gen_661 = torch.utils.data.DataLoader(train_dataset_661, batch_size = 128, shuffle = False)
test_dataset_661 = TimeseriesDataset(agg_p_test_661,label_test_661,seq_len=20) 
test_gen_661 = torch.utils.data.DataLoader(test_dataset_661, batch_size = 128, shuffle = False)  


# In[ ]:


def get_parameters_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.') 


# In[ ]:

import torchinfo
torchinfo.summary(model,input_size=(128,20,1)) 


# In[ ]:


def prune_net(net,prune_amount = 0.2):
    """Prune 100*p% net's weights that have abs(value) approx. 0
    Function that will be use when an iteration is reach
    Args:

    Return:
        newnet (nn.Module): a newnet contain mask that help prune network's weight
    """
    if not isinstance(net,nn.Module):
        print('Invalid input. Must be nn.Module')
        return
    newnet = copy.copy(net)
    modules_list = []

    for name, module in newnet.named_modules():
        if isinstance(module, torch.nn.Conv1d):
            modules_list += [(module,'weight'),(module,'bias')]
        if isinstance(module, torch.nn.Linear):
            modules_list += [(module,'weight'),(module,'bias')]
        if isinstance(module, torch.nn.GroupNorm):
            modules_list += [(module,'weight'),(module,'bias')]
        if isinstance(module, torch.nn.LSTM):
            for name, _ in module.named_parameters():
                if "weight" in name:
                    modules_list +=[(module, name)]
                if "bias" in name:
                    modules_list +=[(module, name)]
    prune.global_unstructured(
        modules_list,
        pruning_method=prune.L1Unstructured,
        amount=prune_amount,)
    return newnet 


# In[ ]:
    
def test_prune_model(model,test_gen):
    
    i = 0
    for data, label in test_gen:
        if i == 0:
            test_data_ = data
            test_label_ = label
            i += 1
        else:
            test_data_ = torch.cat([test_data_, data], dim=0)
            test_label_ = torch.cat([test_label_, label], dim=0)
    test_data_ = test_data_.type(torch.FloatTensor)
    test_label_ = test_label_.type(torch.FloatTensor).numpy()
    
    print(f"The test data length is : {test_data_.shape}")
    
    time_record = []
    ratio_record = []
    f1_record = []
    acc_record = []
    recall_record = []
    precision_record = [] 
    
    with torch.no_grad():
        
        start_time = time.perf_counter()
        _ = model(test_data_)
        time_cost = time.perf_counter() - start_time
        _ = _.detach().numpy()
        print("Orginal Model",time_cost)
        pred_label = get_prediction(_)
        f1,acc,recall,precison = calculate_metrics(pred_label, test_label_, return_ = True)
        time_record.append(time_cost)
        ratio_record.append(0)
        f1_record.append(f1)
        acc_record.append(acc)
        recall_record.append(recall)
        precision_record.append(precison)
        del _
        
        for ratio in np.linspace(0.1, 0.9, 9, endpoint=True):
            
            model_p = prune_net(model,prune_amount=ratio)
            start_time = time.perf_counter()
            _ = model_p(test_data_)
            time_cost = time.perf_counter() - start_time
            _ = _.detach().numpy()
            print(f"Prune {'%.2f'%ratio} Model",time_cost)
            pred_label = get_prediction(_)
            f1,acc,recall,precison = calculate_metrics(pred_label, test_label_, return_ = True)
            time_record.append(time_cost)
            ratio_record.append(ratio)
            f1_record.append(f1) 
            acc_record.append(acc)
            recall_record.append(recall)
            precision_record.append(precison)  
            del _
    
        record_dict = dict()
        record_dict['ratio'] = ratio_record
        record_dict['time'] = time_record
        record_dict['f1'] = f1_record
        record_dict['acc'] = acc_record
        record_dict['recall'] = recall_record
        record_dict['pre'] = precision_record
    return record_dict 


# In[ ]:


outcome = test_prune_model(model,test_gen_661)


# In[ ]:

fig,ax1 = plt.subplots(figsize=(6,6))
ax2 = ax1.twinx()      
ax1.plot(outcome['ratio'],outcome['time'],label = "Time Cost", color ='red')
ax2.plot(outcome['ratio'],outcome['f1'],label = 'F1-Score',color = 'green')
ax1.legend(loc = 'lower left')
ax2.legend(loc = 'upper right')
ax1.set_ylabel('Time Cost: second')   
ax2.set_ylabel('F1-Score') 
ax1.set_xlabel("Pruning Ratio")
plt.grid() 
plt.title("Model Performance VS Compression")
plt.savefig('./model_prune.pdf',bbox_inches = 'tight',pad_inches = 0.1)
plt.show() 

