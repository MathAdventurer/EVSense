#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: Wang,Xudong 220041020 SDS time:2021/7/15


import numpy as np
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

##Self Implement package
from ..data_processing import TimeseriesDataset, get_resident_dt, data_scalar, train_test_data_split 
from ..utils import * 
from ..session_analysis import *
from ..model.models import EV_detect_net, EV_detect_net_1, EV_detect_seq2point, EV_detect_seq2point_BiLSTM, EV_detect_seq2point_BiLSTM_LN,EVsense_dummy_network,EVsense_DNN
from ..experiment import experiment, experiment_lr
from ..model.metrics import *   
from ..model.loss import * 

## Set pandas dataFrame to show all the columns 
pd.set_option('display.max_columns', None)

## Fix random state
setup_seed(0)  

with open('../pickle_data/3000.pkl','rb') as f:
       dt_3000_1min = pickle.load(f) 


time_split_3000 = {'train':{'start':'2018-05-01','end':'2018-06-30'},
              'test':{'start':'2018-07-01','end':'2018-07-31'}}
agg_p_train_3000, label_train_3000, agg_p_test_3000, label_test_3000 = train_test_data_split(dt_3000_1min,time_split_3000)


 ## Load data for Pytorch
train_dataset_3000 = TimeseriesDataset(agg_p_train_3000,label_train_3000,seq_len= 20)
train_gen_3000 = torch.utils.data.DataLoader(train_dataset_3000, batch_size = 128, shuffle = False)
test_dataset_3000 = TimeseriesDataset(agg_p_test_3000,label_test_3000,seq_len=20)
test_gen_3000 = torch.utils.data.DataLoader(test_dataset_3000, batch_size = 128, shuffle = False)


## transfer
trans_f_model = torch.load('../global-state_from661.pth')
trans_f_model = get_transfer_model(trans_f_model,reparam=False,cuda=True,verbose=False)

predict_new_output(trans_f_model,test_gen=test_gen_3000, seq_len = 20, ofilter=None ,cuda = True)

trans_f_model = get_transfer_model(trans_f_model,reparam=True,cuda=True,verbose=False)

train_record_3000_trans_f, train_pred_record_dict_3000_trans_f, test_record_3000_trans_f,test_pred_record_dict_3000_trans_f = experiment_lr(trans_f_model,
                                                                                                                                            train_model=True,
                                                                                                                                            params={'epoch':200,
                                                                                                                                                     'lr':0.001,
                                                                                                                                                     'weight_decay':0.0005,
                                                                                                                                                     'use_cuda':True,
                                                                                                                                                     'saved':True},
                                                                                                                                            train_loader=train_gen_3000,
                                                                                                                                            test_loader=test_gen_3000)



find_the_best(train_record_3000_trans_f,test_record_3000_trans_f,epoch_num=200,resident_id=1642)


summer_3000_outcome_trans_f = visuliaztion_summary_prediction(epoch_num=194,
                                    agg_p_train = agg_p_train_3000,
                                    agg_p_test = agg_p_test_3000,
                                    label_train = label_train_3000,
                                    label_test = label_test_3000,
                                    train_prediction = train_record_3000_trans_f,
                                    test_prediction = test_record_3000_trans_f,
                                    resident_id = 1642,
                                    seq_len = 20,
                                    train_s_t = None,
                                    test_s_t = None,
                                    ofilter = [28,20],
                                    saved = False,
                                   cuda = True)