#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# time:2021/7/11

import os 
import time 
import datetime
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import prune
torch.set_default_tensor_type(torch.FloatTensor)
np.random.seed(0)
torch.manual_seed(0)
import pickle5 as pickle
from sklearn.metrics import accuracy_score, recall_score, precision_score,f1_score

print(os.getcwd())
print(os.listdir())


def get_parameters_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params} total parameters')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params} training parameters') 

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

    
def calculate_metrics(pred, ture, return_ = False):
    """
    :param pred: numpy
    :param ture: numpy
    :return: None
    """
    f1 = f1_score(pred, ture)
    acc = accuracy_score(pred, ture)
    recall = recall_score(pred, ture)
    precision = precision_score(pred, ture)
    
    print("F1-score", f1)
    print("Acc", acc)
    print("Recall",  recall)
    print("Precision", precision)
    
    return f1,acc,recall,precision 
    
def get_prediction(test_prediction):  

    temp = test_prediction.copy()
    temp[np.where(temp > 0.5)] = 1
    temp[np.where(temp <= 0.5)] = 0
    return temp
   
    
def test_prune_model(model,test_data_, test_label_):
    

    test_data_ = torch.tensor(test_data_).type(torch.FloatTensor)

    
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
        
        with open('./experiment_record_prune_model.pkl','wb') as f:
            pickle.dump(record_dict,f)
            print("Saved Done!")
    return record_dict

with open('./test_data_.pkl','rb') as f:
    test_data_ = pickle.load(f)
with open('./test_label_.pkl','rb') as f:
    test_label_ = pickle.load(f)

model = torch.load('../../global-state_from661.pth',map_location=torch.device('cpu'))
outcome = test_prune_model(model,test_data_,test_label_)

