#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# time:2021/7/15


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


 ## Load data for Pytorch and resample data to 900Hz
 
train_dataset_3000 = TimeseriesDataset(agg_p_train_3000[::15],label_train_3000[::15],seq_len= 20)
train_gen_3000 = torch.utils.data.DataLoader(train_dataset_3000, batch_size = 128, shuffle = False)
test_dataset_3000 = TimeseriesDataset(agg_p_test_3000[::15],label_test_3000[::15],seq_len=20) 
test_gen_3000 = torch.utils.data.DataLoader(test_dataset_3000, batch_size = 128, shuffle = False)  


plt.figure()
plt.plot(agg_p_train_3000[::15])
plt.plot(label_train_3000[::15])
plt.plot(label_train_3000[::15]*agg_p_train_3000[::15])
plt.show() 

## Here the transfer model can from self or other residents with sampling rate 60Hz

trans_f_model = torch.load('../global-state_from661.pth')

# model_trans_splr_sum_20 = get_transfer_model(model_3000_original_summer, reparam = False, cuda = True, verbose = False)

model_trans_splr_sum_20 = get_transfer_model(trans_f_model, reparam = False, cuda = True, verbose = False)

predict_new_output(model_trans_splr_sum_20,test_gen=test_gen_3000, seq_len = 20, ofilter=[28,25] ,cuda = True) 

model_trans_splr_sum_20 =  EVsense_DNN(sequence_length=20,cuda=True,hidden_layer_dropout=0.2)

train_record_3000_tran_spl_20, train_pred_record_dict_3000_tran_spl_20, test_record_3000_tran_spl_20,test_pred_record_dict_3000_tran_spl_20 = experiment_lr(model_trans_splr_sum_20,
                                                                                                                                            train_model=True,
                                                                                                                                            params={'epoch':200,
                                                                                                                                                     'lr':0.001,
                                                                                                                                                     'weight_decay':0.0005,
                                                                                                                                                     'use_cuda':True,
                                                                                                                                                     'saved':True},
                                                                                                                                            train_loader=train_gen_3000,
                                                                                                                                            test_loader=test_gen_3000)    
find_the_best(train_record_3000_tran_spl_20,test_record_3000_tran_spl_20)



summer_3000_outcome_tran_spl_20 = visuliaztion_summary_prediction(epoch_num=107, 
                                    agg_p_train = agg_p_train_3000[::15], 
                                    agg_p_test = agg_p_test_3000[::15], 
                                    label_train = label_train_3000[::15], 
                                    label_test = label_test_3000[::15], 
                                    train_prediction = train_pred_record_dict_3000_tran_spl_20, 
                                    test_prediction = test_pred_record_dict_3000_tran_spl_20, 
                                    resident_id = 3000, 
                                    seq_len = 20, 
                                    train_s_t = None,
                                    test_s_t = None,
                                    ofilter = [2,1],
                                    saved = False,
                                   cuda = True) 