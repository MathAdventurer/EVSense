#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: Wang,Xudong 220041020 SDS time:2021/6/22
# The Chinese University of Hongkong, Shenzhen

# Please noted that the hmmlearn package is required.
import sys
import json
import scipy as sp
from scipy import signal
from hmmlearn.hmm import GaussianHMM
from collections import OrderedDict
from copy import deepcopy
import itertools
import pickle 
from ..utils import calculate_metrics
import numpy as np 

# Function to sort parameters from its lowest state
def return_sorting_mapping(means):
    means_copy = deepcopy(means)
    means_copy = np.sort(means_copy, axis=0)

    # Finding mapping
    mapping = {}
    for i, val in enumerate(means_copy):
        mapping[i] = np.where(val == means)[0][0]
    return mapping


def sort_startprob(mapping, startprob):
    num_elements = len(startprob)
    new_startprob = np.zeros(num_elements)
    for i in range(len(startprob)):
        new_startprob[i] = startprob[mapping[i]]
    return new_startprob


def sort_covars(mapping, covars):
    new_covars = np.zeros_like(covars)
    for i in range(len(covars)):
        new_covars[i] = covars[mapping[i]]
    return new_covars


def sort_transition_matrix(mapping, A):
    num_elements = len(A)
    A_new = np.zeros((num_elements, num_elements))
    for i in range(num_elements):
        for j in range(num_elements):
            A_new[i, j] = A[mapping[i], mapping[j]]
    return A_new


def sort_learnt_parameters(startprob, means, covars, transmat):
    mapping = return_sorting_mapping(means)
    means_new = np.sort(means, axis=0)
    startprob_new = sort_startprob(mapping, startprob)
    covars_new = sort_covars(mapping, covars)
    transmat_new = sort_transition_matrix(mapping, transmat)
    assert np.shape(means_new) == np.shape(means)
    assert np.shape(startprob_new) == np.shape(startprob)
    assert np.shape(transmat_new) == np.shape(transmat)

    return [startprob_new, means_new, covars_new, transmat_new]


# Function to combined each parameter off all appliances
def compute_A_fhmm(list_A):
    result = list_A[0]
    for i in range(len(list_A) - 1):
        result = np.kron(result, list_A[i + 1])
    return result


def compute_pi_fhmm(list_pi):
    result = list_pi[0]
    for i in range(len(list_pi) - 1):
        result = np.kron(result, list_pi[i + 1])
    return result


def compute_means_fhmm(list_means):
    states_combination = list(itertools.product(*list_means))
    num_combinations = len(states_combination)
    means_stacked = np.array([sum(x) for x in states_combination])
    means = np.reshape(means_stacked, (num_combinations, 1))
    cov = np.tile(5 * np.identity(1), (num_combinations, 1, 1))

    return [means, cov]


def create_combined_hmm(model):
    list_pi = [model[appliance].startprob_ for appliance in model]
    list_A = [model[appliance].transmat_ for appliance in model]
    list_means = [model[appliance].means_.flatten().tolist()
                  for appliance in model]

    pi_combined = compute_pi_fhmm(list_pi)
    A_combined = compute_A_fhmm(list_A)
    [mean_combined, cov_combined] = compute_means_fhmm(list_means)

    combined_model = GaussianHMM(n_components=len(pi_combined), covariance_type='full')
    combined_model.startprob_ = pi_combined
    combined_model.transmat_ = A_combined
    combined_model.covars_ = cov_combined
    combined_model.means_ = mean_combined

    return combined_model


# Disagregate energy, finding state and power of each appliance
def decode_hmm(length_sequence, centroids, appliance_list, states):
    hmm_states = OrderedDict()
    hmm_power = OrderedDict()
    total_num_combinations = 1

    for appliance in appliance_list:
        total_num_combinations *= len(centroids[appliance])

    for appliance in appliance_list:
        hmm_states[appliance] = np.zeros(length_sequence, dtype=np.int)
        hmm_power[appliance] = np.zeros(length_sequence)

    for i in range(length_sequence):
        factor = total_num_combinations

        for appliance in appliance_list:
            factor = factor // len(centroids[appliance])
            temp = int(states[i]) / factor
            hmm_states[appliance][i] = temp % len(centroids[appliance])
            hmm_power[appliance][i] = centroids[
                appliance][hmm_states[appliance][i]]

    return [hmm_states, hmm_power]


def disaggregate(combined_model, model, test_power):
    length = len(test_power)
    temp = test_power.reshape(length, 1)
    states = (combined_model.predict(temp))

    means = OrderedDict()
    for appliance in model:
        means[appliance] = (
            model[appliance].means_.flatten().tolist())
        means[appliance].sort()

    [decoded_states, decoded_power] = decode_hmm(
        len(states), means, means.keys(), states)

    return [decoded_power, decoded_states]


## Benchmark of FHMM model for resident 3000
## Here think the EV charging with 2 state and 4 state for agggate power: NoEV low, NoEV high, HaveEV low, HaveEV high.
## Disaggregate and extract the EV charging state from the aggregate power.

if __name__ == '__main__':
    
    with open('../pickle_data/3000.pkl', 'rb') as f:
        dt_3000_1min = pickle.load(f)
        
    dt_3000_1min['No_EV_agg_p'] = dt_3000_1min['aggregate'] - dt_3000_1min['car1']
    ev_train = dt_3000_1min.loc['2018-05-01':'2018-06-30']['car1'].values.reshape(-1, 1)
    ev_test = dt_3000_1min.loc['2018-07-01':'2018-07-31']['car1'].values.reshape(-1, 1)
    label_train_hmm = dt_3000_1min.loc['2018-05-01':'2018-06-30']['label'].values.reshape(-1, 1)
    agg_1_train = dt_3000_1min.loc['2018-05-01':'2018-06-30']['No_EV_agg_p'].values.reshape(-1, 1)
    agg_1_test = dt_3000_1min.loc['2018-07-01':'2018-07-31']['No_EV_agg_p'].values.reshape(-1, 1)
    label_test_hmm = dt_3000_1min.loc['2018-07-01':'2018-07-31']['label'].values.reshape(-1, 1)

    assert ev_train.shape == label_train_hmm.shape == agg_1_train.shape
    assert ev_test.shape == label_test_hmm.shape == agg_1_test.shape

    train_power = OrderedDict()
    train_power['EV'] = ev_train
    train_power['agg'] = agg_1_train
    num_appliances = len(train_power)

    state_appliances = OrderedDict()
    state_appliances['EV'] = 2
    state_appliances['agg'] = 4
    model = OrderedDict()
    for appliance in train_power:
        model[appliance] = GaussianHMM(n_components=state_appliances[appliance], covariance_type="full",
                                    random_state=0).fit(train_power[appliance])

    startprob = OrderedDict()
    for appliance in train_power:
        if state_appliances[appliance] == 2:
            startprob[appliance] = [0.5, 0.5]
        elif state_appliances[appliance] == 3:
            startprob[appliance] = [1 / 3, 1 / 3, 1 / 3]
        else:
            startprob[appliance] = model[appliance].startprob_.tolist()
    # startprob[appliance] = model[appliance].startprob_.tolist()

    buff = "{"

    for appliance in model:
        buff += "\"" + appliance + "\" : {" + \
                "\"startprob\" : " + str(startprob[appliance]) + \
                ", \"transmat\" : " + str(model[appliance].transmat_.tolist()) + \
                ",\"means\" : " + str(model[appliance].means_.tolist()) + \
                ", \"covars\" : " + str(model[appliance].covars_.tolist()) + "},"

    buff = buff[0:len(buff) - 1]
    buff += "}"
    print(buff)

    data_json = json.loads(buff)
    transmat = OrderedDict()
    means = OrderedDict()
    covars = OrderedDict()

    for appliance in list(data_json.keys()):
        transmat[appliance] = np.array(data_json[appliance]['transmat'])
        means[appliance] = np.array(data_json[appliance]['means'])
        covars[appliance] = np.array(data_json[appliance]['covars'])

    new_model = OrderedDict()
    for appliance in model:
        startprob_new, means_new, covars_new, transmat_new = sort_learnt_parameters(
            startprob[appliance], means[appliance],
            covars[appliance], transmat[appliance])

        new_model[appliance] = GaussianHMM(n_components=startprob_new.size, covariance_type="full", random_state=0)
        new_model[appliance].startprob_ = startprob_new
        new_model[appliance].transmat_ = transmat_new
        new_model[appliance].means_ = means_new
        new_model[appliance].covars_ = covars_new

    combined_model = create_combined_hmm(new_model)

    train_power = agg_1_train + ev_train
    test_power = agg_1_test + ev_test

    length = len(train_power)
    [predicted_power, predicted_states] = disaggregate(combined_model, new_model, train_power)
    predicted_total = np.zeros(length)
    for appliance in predicted_power:
        predicted_power[appliance] = predicted_power[appliance]
        predicted_total += predicted_power[appliance]

    print("Train: \n")
    print("Oringal state:")
    calculate_metrics(predicted_states['EV'], label_train_hmm)
    print("Inverse state:")
    calculate_metrics(1 - predicted_states['EV'], label_train_hmm)

    length = len(test_power)
    [predicted_power, predicted_states] = disaggregate(combined_model, new_model, test_power)
    predicted_total = np.zeros(length)
    for appliance in predicted_power:
        predicted_power[appliance] = predicted_power[appliance]
        predicted_total += predicted_power[appliance]

    print("Test: \n")
    print("Oringal state:")
    calculate_metrics(predicted_states['EV'], label_test_hmm)
    print("Inverse state:")
    calculate_metrics(1 - predicted_states['EV'], label_test_hmm)
