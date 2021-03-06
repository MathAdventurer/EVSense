#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# EVSense: Xudong Wang, Guoming Tang
# time:2021/5/29

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import random
import sys
sys.path.append(".") 
from .loss import CORAL

# hidden_layer_dropout = 0.2 # we can change the dropout rate

def set_seed(seed=0):
    """
    The function is used to set the random seed for the neural network.
    Please noted that the default seed is 0.
    You can use this function when initialization some neural network.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def dummy_network(sequence_length, hidden_layer_dropout, cuda):
    """
    Update for the initialization, delete the input parameter "cuda"
    The Uniform def the device use for the experiment file.
    Update 0610, still use the cuda option.
    """

    # Model architecture
    set_seed()

    model = torch.nn.Sequential(

        torch.nn.Conv1d(out_channels=30, kernel_size=3, in_channels=1),
        torch.nn.ReLU(),

        torch.nn.Conv1d(out_channels=30, kernel_size=5, in_channels=30),
        torch.nn.ReLU(),

        torch.nn.Conv1d(out_channels=40, kernel_size=6, in_channels=30),
        torch.nn.ReLU(),

        torch.nn.Conv1d(out_channels=50, kernel_size=5, in_channels=40),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=hidden_layer_dropout),

        torch.nn.Conv1d(out_channels=50, kernel_size=5, in_channels=50),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=hidden_layer_dropout),

        torch.nn.Flatten(),
    )
    if cuda:
        model.cuda()
    return model


def EVsense_dummy_network(sequence_length, hidden_layer_dropout, cuda):
    """
    Update for final use the paper
    Update for the initialization, delete the input parameter "cuda"
    The Uniform def the device use for the experiment file.
    Update 0610, still use the cuda option.
    """

    # Model architecture
    set_seed()

    if sequence_length == 20:

        model = torch.nn.Sequential(

            torch.nn.Conv1d(out_channels=30, kernel_size=3, in_channels=1),
            torch.nn.ReLU(),

            torch.nn.Conv1d(out_channels=30, kernel_size=5, in_channels=30),
            torch.nn.ReLU(),

            torch.nn.Conv1d(out_channels=40, kernel_size=6, in_channels=30),
            torch.nn.ReLU(),

            torch.nn.Conv1d(out_channels=50, kernel_size=5, in_channels=40),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=hidden_layer_dropout),

            torch.nn.Conv1d(out_channels=50, kernel_size=5, in_channels=50),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=hidden_layer_dropout),

            torch.nn.Flatten(),
        )
    elif sequence_length == 10:

        model = torch.nn.Sequential(

            torch.nn.Conv1d(out_channels=30, kernel_size=2, in_channels=1),
            torch.nn.ReLU(),

            torch.nn.Conv1d(out_channels=30, kernel_size=3, in_channels=30),
            torch.nn.ReLU(),

            torch.nn.Conv1d(out_channels=40, kernel_size=3, in_channels=30),
            torch.nn.ReLU(),

            torch.nn.Conv1d(out_channels=50, kernel_size=4, in_channels=40),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=hidden_layer_dropout),

            torch.nn.Conv1d(out_channels=50, kernel_size=2, in_channels=50),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=hidden_layer_dropout),

            torch.nn.Flatten(),
        )
    else:
        print("Not support input sequence length, please use length 10 or 20.")

    if cuda:
        model.cuda()

    return model


class EVsense_DNN(nn.Module):

    def __init__(self, sequence_length, hidden_layer_dropout, cuda):

        set_seed()
        super(EVsense_DNN, self).__init__()
        dummy_model = EVsense_dummy_network(sequence_length, hidden_layer_dropout, cuda)
        rand_tensor = torch.randn(1, 1, sequence_length).type(torch.FloatTensor)

        if cuda:
            rand_tensor = rand_tensor.to(device='cuda')

        dummy_output = dummy_model(rand_tensor)
        self.sequence_length = sequence_length
        self.hidden_layer_dropout = hidden_layer_dropout
        self.num_of_flattened_neurons = dummy_output.shape[-1]

        if self.sequence_length == 20:
            self.conv1 = torch.nn.Conv1d(out_channels=30, kernel_size=3, in_channels=1)
            self.conv2 = torch.nn.Conv1d(out_channels=30, kernel_size=5, in_channels=30)
            self.conv3 = torch.nn.Conv1d(out_channels=40, kernel_size=6, in_channels=30)
            self.conv4 = torch.nn.Conv1d(out_channels=50, kernel_size=5, in_channels=40)
            self.conv5 = torch.nn.Conv1d(out_channels=50, kernel_size=5, in_channels=50)
        else:
            self.conv1 = torch.nn.Conv1d(out_channels=30, kernel_size=2, in_channels=1)
            self.conv2 = torch.nn.Conv1d(out_channels=30, kernel_size=3, in_channels=30)
            self.conv3 = torch.nn.Conv1d(out_channels=40, kernel_size=3, in_channels=30)
            self.conv4 = torch.nn.Conv1d(out_channels=50, kernel_size=4, in_channels=40)
            self.conv5 = torch.nn.Conv1d(out_channels=50, kernel_size=2, in_channels=50)

        self.lstm = nn.LSTM(input_size=self.num_of_flattened_neurons, hidden_size=50, num_layers=2, bidirectional=True,
                            batch_first=True, dropout=0.1)

        self.fc1 = torch.nn.Linear(out_features=1024, in_features=50 * 2)
        self.fc2 = torch.nn.Linear(out_features=1, in_features=1024)
        self.dropout1 = torch.nn.Dropout(self.hidden_layer_dropout)
        self.dropout2 = torch.nn.Dropout(self.hidden_layer_dropout)

        self.gn1 = nn.GroupNorm(1, 30)
        self.gn2 = nn.GroupNorm(1, 30)
        self.gn3 = nn.GroupNorm(1, 40)
        self.gn4 = nn.GroupNorm(1, 50)
        self.gn5 = nn.GroupNorm(1, 50)

        self.gn6 = nn.GroupNorm(1, 100)
        self.gn7 = nn.GroupNorm(1, 1024)

        if cuda:
            self.cuda()

    def forward(self, x):

        assert x.shape[1] == self.sequence_length
        assert x.shape[2] == 1

        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.gn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.gn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.gn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.gn4(x)
        x = F.relu(x)

        x = self.dropout1(x)

        x = self.conv5(x)
        x = self.gn5(x)
        x = F.relu(x)
        x = self.dropout2(x)

        #         print(x.shape) # torch.Size([128, 50, 1])
        x = x.permute(0, 2, 1)
        #         print(x.shape) # torch.Size([128, 1, 50])
        x = self.lstm(x)[0]
        x = x.permute(0, 2, 1)
        x = x.squeeze()
        x = self.gn6(x)

        x = self.fc1(x)
        x = self.gn7(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.sigmoid(x)

        return x


class EVSense_transferable(nn.Module):

    def __init__(self, sequence_length, hidden_layer_dropout, cuda):

        set_seed()
        super(EVSense_transferable, self).__init__()
        dummy_model = EVsense_dummy_network(sequence_length, hidden_layer_dropout, cuda)
        rand_tensor = torch.randn(1, 1, sequence_length).type(torch.FloatTensor)

        if cuda:
            rand_tensor = rand_tensor.to(device='cuda')

        dummy_output = dummy_model(rand_tensor)
        self.sequence_length = sequence_length
        self.hidden_layer_dropout = hidden_layer_dropout
        self.num_of_flattened_neurons = dummy_output.shape[-1]
#         self.transfer = transfer

        if self.sequence_length == 20:
            self.conv1 = torch.nn.Conv1d(out_channels=30, kernel_size=3, in_channels=1)
            self.conv2 = torch.nn.Conv1d(out_channels=30, kernel_size=5, in_channels=30)
            self.conv3 = torch.nn.Conv1d(out_channels=40, kernel_size=6, in_channels=30)
            self.conv4 = torch.nn.Conv1d(out_channels=50, kernel_size=5, in_channels=40)
            self.conv5 = torch.nn.Conv1d(out_channels=50, kernel_size=5, in_channels=50)
        else:
            self.conv1 = torch.nn.Conv1d(out_channels=30, kernel_size=2, in_channels=1)
            self.conv2 = torch.nn.Conv1d(out_channels=30, kernel_size=3, in_channels=30)
            self.conv3 = torch.nn.Conv1d(out_channels=40, kernel_size=3, in_channels=30)
            self.conv4 = torch.nn.Conv1d(out_channels=50, kernel_size=4, in_channels=40)
            self.conv5 = torch.nn.Conv1d(out_channels=50, kernel_size=2, in_channels=50)

        self.lstm = nn.LSTM(input_size=self.num_of_flattened_neurons, hidden_size=50, num_layers=2, bidirectional=True,
                            batch_first=True, dropout=0.1)

        self.fc1 = torch.nn.Linear(out_features=1024, in_features=50 * 2)
        self.fc2 = torch.nn.Linear(out_features=1, in_features=1024)
        self.dropout1 = torch.nn.Dropout(self.hidden_layer_dropout)
        self.dropout2 = torch.nn.Dropout(self.hidden_layer_dropout)

        self.gn1 = nn.GroupNorm(1, 30)
        self.gn2 = nn.GroupNorm(1, 30)
        self.gn3 = nn.GroupNorm(1, 40)
        self.gn4 = nn.GroupNorm(1, 50)
        self.gn5 = nn.GroupNorm(1, 50)

        self.gn6 = nn.GroupNorm(1, 100)
        self.gn7 = nn.GroupNorm(1, 1024)
        

        if cuda:
            self.cuda()

    def forward(self, x, transfer = False, x_source = None, domain_loss = CORAL, FC12_weight = None):
        """
        transfer: bool, True means doing the transfer learning, then x_source data, domain_loss is required
        FC12_weight: List [w_1,w_2] for CORAL loss of FC1 and FC2, if None, are equal weight 1.
        """
        if transfer:
            
            assert x.shape[1] == self.sequence_length
            assert x.shape[2] == 1

            x = x.permute(0, 2, 1)

            x = self.conv1(x)
            x = self.gn1(x)
            x = F.relu(x)

            x = self.conv2(x)
            x = self.gn2(x)
            x = F.relu(x)

            x = self.conv3(x)
            x = self.gn3(x)
            x = F.relu(x)

            x = self.conv4(x)
            x = self.gn4(x)
            x = F.relu(x)

            x = self.dropout1(x)

            x = self.conv5(x)
            x = self.gn5(x)
            x = F.relu(x)
            x = self.dropout2(x)

            #         print(x.shape) # torch.Size([128, 50, 1])
            x = x.permute(0, 2, 1)
            #         print(x.shape) # torch.Size([128, 1, 50])
            x = self.lstm(x)[0]
            x = x.permute(0, 2, 1)
            x = x.squeeze()
            x = self.gn6(x)
                 
            
            x_source = x_source.permute(0, 2, 1)

            x_source = self.conv1(x_source)
            x_source = self.gn1(x_source)
            x_source = F.relu(x_source)

            x_source = self.conv2(x_source)
            x_source = self.gn2(x_source)
            x_source = F.relu(x_source)

            x_source = self.conv3(x_source)
            x_source = self.gn3(x_source)
            x_source = F.relu(x_source)

            x_source = self.conv4(x_source)
            x_source = self.gn4(x_source)
            x_source = F.relu(x_source)

            x_source = self.dropout1(x_source)

            x_source = self.conv5(x_source)
            x_source = self.gn5(x_source)
            x_source = F.relu(x_source)
            x_source = self.dropout2(x_source)

            #         print(x_source.shape) # torch.Size([128, 50, 1])
            x_source = x_source.permute(0, 2, 1)
            #         print(x_source.shape) # torch.Size([128, 1, 50])
            x_source = self.lstm(x_source)[0]
            x_source = x_source.permute(0, 2, 1)
            x_source = x_source.squeeze()
            x_source = self.gn6(x_source)
            
            
            ## Transfer the fc-1 
            x = self.fc1(x)
            x_source = self.fc1(x_source) 
            transfer_loss_1 = CORAL(x_source,x)
                                    
            x = self.gn7(x)
            x = F.relu(x)
                                    
            x_source = self.gn7(x_source)
            x_source = F.relu(x_source)

            ## Transfer the fc-2
            x = self.fc2(x)           
            x_source = self.fc2(x_source)
            transfer_loss_2 = CORAL(x_source,x)
                                    
            x = F.sigmoid(x)
            x_source = F.sigmoid(x_source)
            
            if FC12_weight:
                transfer_loss_sum = FC12_weight[0]*transfer_loss_1 + FC12_weight[1]*transfer_loss_2
            else:
                transfer_loss_sum = transfer_loss_1 + transfer_loss_2
            
            
            return transfer_loss_sum
            
        else:
            assert x.shape[1] == self.sequence_length
            assert x.shape[2] == 1

            x = x.permute(0, 2, 1)

            x = self.conv1(x)
            x = self.gn1(x)
            x = F.relu(x)

            x = self.conv2(x)
            x = self.gn2(x)
            x = F.relu(x)

            x = self.conv3(x)
            x = self.gn3(x)
            x = F.relu(x)

            x = self.conv4(x)
            x = self.gn4(x)
            x = F.relu(x)

            x = self.dropout1(x)

            x = self.conv5(x)
            x = self.gn5(x)
            x = F.relu(x)
            x = self.dropout2(x)

            #         print(x.shape) # torch.Size([128, 50, 1])
            x = x.permute(0, 2, 1)
            #         print(x.shape) # torch.Size([128, 1, 50])
            x = self.lstm(x)[0]
            x = x.permute(0, 2, 1)
            x = x.squeeze()
            x = self.gn6(x)

            x = self.fc1(x)
            x = self.gn7(x)
            x = F.relu(x)

            x = self.fc2(x)
            x = F.sigmoid(x)

            return x
