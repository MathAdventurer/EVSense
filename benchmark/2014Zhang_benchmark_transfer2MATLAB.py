#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# time:2021/7/18
import os
import pickle
import numpy as np
import scipy.io as io
import sys 
sys.path.append("..") 
from ..utils import calculate_metrics, out_filter

# Handle data in Python and saved for MATLAB use. 
with open('../pickle_data/3000.pkl', 'rb') as f:
    dt_3000_1min = pickle.load(f)

useto2014 = dt_3000_1min.loc['2018-07-01':'2018-07-31']['aggregate'].values[:1440*25] * 1000
useto2014_label = dt_3000_1min.loc['2018-07-01':'2018-07-31']['label'].values[:1440*25]
saved2014 = np.zeros((1440, 25))
useto2014[np.isnan(useto2014)] = 0
useto2014_label[np.isnan(useto2014_label)] = 0
for i in range(saved2014.shape[1]):
    saved2014[:, i] = useto2014[0 + i * 1440:1440 + i * 1440]

mat_path = '../agg_signal_3000.mat'
io.savemat(mat_path, {'agg_signal_3000': saved2014})
## Current data can be directly run in the Benchmark Zhang's Model 
## Training-Free Non-Intrusive Load Monitoring of Electric Vehicle Charging with Low Sampling Rate
## https://www.mathworks.com/matlabcentral/fileexchange/47474-energy-disaggregation-algorithm-for-electric-vehicle-charging-load


## Please save the MATLAB outcome in mat formula and using the following command to calculate metrics
## e.g save the MATLAB outcome as prediction_3000.mat. 
 if not os.path.exists(f"../prediction_3000.mat"):
      print("Not found Matlab prediction outcome: 'prediction_3000.mat' /n Please save the MATLAB outcome in mat formula to calculate metrics!")
      raise IOError('File Not Exist.') 

ev2014pred_3000 = io.loadmat('../prediction_3000.mat')['EVsignal_3000']
ev2014pred_3000_ = np.zeros_like(useto2014_label)
for i in range(ev2014pred_3000.shape[1]):
    ev2014pred_3000_[0 + i * 1440:1440 + i * 1440] = ev2014pred_3000[:, i]
ev2014pred_3000_[ev2014pred_3000_ > 0] = 1
calculate_metrics(ev2014pred_3000_.reshape(-1, 1), useto2014_label.reshape(-1, 1))
calculate_metrics(out_filter(ev2014pred_3000_.reshape(-1, 1), 60, 60), useto2014_label.reshape(-1, 1))
