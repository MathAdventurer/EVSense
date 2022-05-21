#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# EVSense: Xudong Wang, Guoming Tang

import os
import sys
sys.path.append("..")
import torch

from ..model.models import EVsense_dummy_network,EVsense_DNN,EVSense_transferable
from ..model_pruning_compression.model_pruning_compression import prune_net, get_transfer_model

def create_model_dict(edge_model_path):
    model_dict = dict()
    current_files = os.listdir(edge_model_path)
    edge_model_list = []

    for file_ in current_files:
        if file_.endswith('.pth') and 'edge' in file_:
            edge_model_list.append(file_)

    print(f'Total {len(edge_model_list)} edge model are found.')

    for i in range(len(edge_model_list)):
        model_name = "edge_model_" + str(i)
        model_info = torch.load(edge_model_path + edge_model_list[i], map_location=torch.device('cpu'))
        model_dict.update({model_name: model_info})

    return model_dict

def get_averaged_weights(model_dict, weights_name):
    edge_model_list = list(model_dict.keys())

    with torch.no_grad():
        for i in range(len(edge_model_list)):
            if i == 0:
                weights_ = model_dict[edge_model_list[i]][weights_name].data.clone()
            weights_ += model_dict[edge_model_list[i]][weights_name].data.clone()

        weights_ = weights_ / (len(edge_model_list))

    return weights_


def get_federated_avg_model(model_dict, cuda, pruning_ratio):
    fed_avg_model = EVSense_transferable(sequence_length=20, cuda=False, hidden_layer_dropout=0.2)

    with torch.no_grad():
        for weights_ in list(fed_avg_model.state_dict().keys()):
            fed_avg_model.state_dict()[weights_] = get_averaged_weights(model_dict, weights_).data.clone()

    if pruning_ratio:
        fed_avg_model = prune_net(fed_avg_model, pruning_ratio)

    fed_avg_model = get_transfer_model(fed_avg_model, reparam=False, cuda=cuda, verbose=False)

    return fed_avg_model