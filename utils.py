#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: Wang,Xudong 220041020 SDS time:2021/5/29
import time
import datetime
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import copy
import matplotlib.pyplot as plt
# fix random state
def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
#     random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True 

# Data I/O Part
def str2time(str_in: str):
    try:
        output = datetime.datetime.strptime(str_in[:19], '%Y-%m-%d %H:%M:%S')  # [:19] correct error in sql query
        return output
    except:
        print("Error timestamp! Please Check!")
        return "_"


def Label_EV_data(df: pd.DataFrame, threshold: float):
    if df['car1'] >= threshold:
        # if df['car1'] >= threshold or df['car2'] >= threshold:  # Not use the car2 data, since the car2 only
        # recorded when the home have two car.
        return 1
    else:
        return 0


def input_filter(aggregate_p_array, threshold: float):
    aggregate_p = aggregate_p_array.copy()
    aggregate_p[np.where(aggregate_p < threshold)] = 0
    return aggregate_p


def input_filter_df(df: pd.DataFrame, threshold):  # input filter for the dataframe which contain the column 'aggregate'

    if df['aggregate'] >= threshold:
        return df['aggregate']
    else:
        return 0


# Get the prediction label
def get_prediction(test_prediction):  # 输入的是numpy (batch,1) 或者 (batch,)

    temp = test_prediction.copy()
    temp[np.where(temp > 0.5)] = 1
    temp[np.where(temp <= 0.5)] = 0
    return temp


def out_filter(test_seq, window_size, threshold):

    test_seq = get_prediction(test_seq)

    stack = []

    stack_count_len = 0
    for i in range(len(test_seq)):
        if test_seq[i] == 0 and len(stack) == 0:
            continue
        elif test_seq[i] == 0 and len(stack) != 0:
            if len(stack) < window_size:
                stack.append(i)
            elif len(stack) >= window_size:
                if stack_count_len >= threshold:
                    test_seq[stack] = 1
                    stack = []
                    stack_count_len = 0
                else:
                    test_seq[stack] = 0
                    stack = []
                    stack_count_len = 0
        elif test_seq[i] == 1 and len(stack) == 0:
            stack.append(i)
            stack_count_len += 1
        elif test_seq[i] == 1 and len(stack) != 0:
            if len(stack) < window_size:
                stack.append(i)
                stack_count_len += 1
            elif len(stack) >= window_size:
                if stack_count_len + 1 >= threshold:
                    test_seq[stack] = 1
                    stack = []
                    stack_count_len = 0
                else:
                    test_seq[stack] = 0
                    stack = []
                    stack_count_len = 0
    return np.array(test_seq)


def data_clean_filter(data, window_size, threshold):

    assert type(data) is np.ndarray
    stack = []
    stack_count_len = 0

    for i in range(len(data)):
        if data[i] == 0 and len(stack) == 0:
            continue
        elif data[i] == 0 and len(stack) != 0:
            if len(stack) < window_size:
                stack.append(i)
            elif len(stack) >= window_size:
                if stack_count_len >= threshold:
                    data[stack] = 1
                    stack = []
                    stack_count_len = 0
                else:
                    data[stack] = 0
                    stack = []
                    stack_count_len = 0
        elif data[i] == 1 and len(stack) == 0:
            stack.append(i)
            stack_count_len += 1
        elif data[i] == 1 and len(stack) != 0:
            if len(stack) < window_size:
                stack.append(i)
                stack_count_len += 1
            elif len(stack) >= window_size:
                if stack_count_len + 1 >= threshold:
                    data[stack] = 1
                    stack = []
                    stack_count_len = 0
                else:
                    data[stack] = 0
                    stack = []
                    stack_count_len = 0
    return np.array(data)




def visualization_record(train_record, test_record, epoch_num, resident_id, saved = False):
    """
    :param train_record: dict
    :param test_record: dict
    :param epoch_num: int
    :param resident_id: int
    :param saved: bool
    :return:
    """
    epoch_list = [i for i in range(1,epoch_num+1)]
    plt.style.use('seaborn')

    plt.figure(figsize=(7.5,5))
    plt.plot(epoch_list, test_record['test_epoch_loss_record'], c ='r', label ='Testing F1-score', markerfacecolor='none', lw= 2)
    plt.plot(epoch_list, train_record['train_epoch_loss_record'], c ='g', label ='Training F1-score', markerfacecolor='none', lw= 2)
    plt.xlabel(r'$epochs$')
    plt.ylabel(r'$metrics$')
    plt.title(f"EVsense-DNN Loss for {epoch_num} epoch, resident: {resident_id}")
    plt.legend(loc = 'best',)
    if saved:
        plt.savefig(f'./{resident_id}_{epoch_num}_loss.pdf')
    plt.show()



    plt.figure(figsize=(16,10))
    plt.subplot(221)
    plt.plot(epoch_list, test_record['test_epoch_f1_record'], c ='r', label ='Testing F1-score', markerfacecolor='none', lw= 2)
    plt.plot(epoch_list, train_record['train_epoch_f1_record'], c ='g', label ='Training F1-score', markerfacecolor='none', lw= 2)
    plt.xlabel(r'$epochs$')
    plt.ylabel(r'$metrics$')
    plt.title(f"EVsense-DNN F1-score for {epoch_num} epoch, resident: {resident_id}")
    plt.legend(loc = 'best',)

    plt.subplot(222)
    plt.plot(epoch_list, test_record['test_epoch_recall_record'], c ='r', label ='Testing F1-score', markerfacecolor='none', lw= 2)
    plt.plot(epoch_list, train_record['train_epoch_recall_record'], c ='g', label ='Training F1-score', markerfacecolor='none', lw= 2)
    plt.xlabel(r'$epochs$')
    plt.ylabel(r'$metrics$')
    plt.title("EVsense-DNN Recall for {epoch_num} epoch")
    plt.legend(loc = 'best',)


    plt.subplot(223)
    plt.plot(epoch_list, test_record['test_epoch_acc_record'], c ='r', label ='Testing F1-score', markerfacecolor='none', lw= 2)
    plt.plot(epoch_list, train_record['train_epoch_acc_record'], c ='g', label ='Training F1-score', markerfacecolor='none', lw= 2)
    plt.xlabel(r'$epochs$')
    plt.ylabel(r'$metrics$')
    plt.title(f"EVsense-DNN Acc for {epoch_num} epoch, resident: {resident_id}")
    plt.legend(loc = 'best',)


    plt.subplot(224)
    plt.plot(epoch_list, test_record['test_epoch_precision_record'], c ='r', label ='Testing F1-score', markerfacecolor='none', lw= 2)
    plt.plot(epoch_list, train_record['train_epoch_precision_record'], c ='g', label ='Training F1-score', markerfacecolor='none', lw= 2)
    plt.xlabel(r'$epochs$')
    plt.ylabel(r'$metrics$')
    plt.title(f"EVsense-DNN Precision for {epoch_num} epoch, resident: {resident_id}")
    plt.legend(loc = 'best',)
    if saved:
        plt.savefig(f'./{resident_id}_{epoch_num}_metrics.pdf')
    plt.show()


def find_the_best(train_record, test_record, epoch_num, resident_id):
    print(f"Resident {resident_id}, Epoch {epoch_num}, Test:")
    print("Best loss (Dice loss)", np.argmin(test_record['test_epoch_loss_record']) + 1)
    print("Best loss F1 score", np.argmax(test_record['test_epoch_f1_record']) + 1)
    print("Best loss Recall", np.argmax(test_record['test_epoch_recall_record']) + 1)
    print("Best loss Precision", np.argmax(test_record['test_epoch_precision_record']) + 1)
    print("Best loss Acc ", np.argmax(test_record['test_epoch_acc_record']) + 1)

    print(f"Resident {resident_id}, Epoch {epoch_num}, Train:")
    print("Best loss (Dice loss)", np.argmin(train_record['train_epoch_loss_record']) + 1)
    print("Best loss F1 score", np.argmax(train_record['train_epoch_f1_record']) + 1)
    print("Best loss Recall", np.argmax(train_record['train_epoch_recall_record']) + 1)
    print("Best loss Precision", np.argmax(train_record['train_epoch_precision_record']) + 1)
    print("Best loss Acc ", np.argmax(train_record['train_epoch_acc_record']) + 1)

    record = [np.argmin(test_record['test_epoch_loss_record']) + 1,
              np.argmax(test_record['test_epoch_f1_record']) + 1,
              np.argmax(test_record['test_epoch_recall_record']) + 1,
              np.argmax(test_record['test_epoch_precision_record']) + 1,
              np.argmax(test_record['test_epoch_precision_record']) + 1,
              np.argmin(train_record['train_epoch_loss_record']) + 1,
              np.argmax(train_record['train_epoch_f1_record']) + 1,
              np.argmax(train_record['train_epoch_recall_record']) + 1,
              np.argmax(train_record['train_epoch_precision_record']) + 1,
              np.argmax(train_record['train_epoch_acc_record']) + 1]
    print(record)
    print([record.count(_) for _ in record])


def calculate_metrics(pred, ture):
    """
    :param pred: numpy
    :param ture: numpy
    :return: None
    """
    print("F1-score", f1_score(pred, ture))
    print("Acc", accuracy_score(pred, ture))
    print("Recall", recall_score(pred, ture))
    print("Precision", precision_score(pred, ture))


def visuliaztion_summary_prediction(epoch_num,
                                    agg_p_train,
                                    agg_p_test,
                                    label_train,
                                    label_test,
                                    train_prediction,
                                    test_prediction,
                                    resident_id,
                                    seq_len,
                                    train_s_t=None,
                                    test_s_t=None,
                                    ofilter=None,
                                    saved=False,
                                    return_=False,
                                    cuda=True):
    """

    :param epoch_num: int
    :param agg_p_train: numpy
    :param agg_p_test: numpy
    :param label_train: numpy
    :param label_test: numpy
    :param train_prediction: record_dict
    :param test_prediction: record_dict
    :param resident_id: int
    :param seq_len: int
    :param train_s_t: list with 2 int element [int, int]
    :param test_s_t: list with 2 int element [int, int]
    :param ofilter: list with two parameters [window, threshold]
    :param saved: boolean
    :param return_: boolean
    :param cuda: boolean
    :return: list [te_pred, label_te, tr_pred, label_tr]
    """
    print("Original length:")
    print(len(agg_p_train), len(label_train))
    print(len(agg_p_test), len(label_test))
    if train_s_t:
        print("Select train period", train_s_t)
        agg_train = agg_p_train[seq_len - 1 + train_s_t[0]:train_s_t[1] + seq_len - 1]
        label_tr = label_train[seq_len - 1 + train_s_t[0]:train_s_t[1] + seq_len - 1]
    else:
        print("Using total train period")
        agg_train = agg_p_train[seq_len - 1:]
        label_tr = label_train[seq_len - 1:]

    print(len(agg_train), len(label_tr))

    if test_s_t:
        print("Select test period", test_s_t)
        agg_test = agg_p_test[seq_len - 1 + test_s_t[0]:test_s_t[1] + seq_len - 1]
        label_te = label_test[seq_len - 1 + test_s_t[0]:test_s_t[1] + seq_len - 1]
    else:
        print("Using total test period")
        agg_test = agg_p_test[seq_len - 1:]
        label_te = label_test[seq_len - 1:]

    print(len(agg_test), len(label_te))

    if cuda:
        tr_pred_pb = train_prediction[epoch_num].detach().numpy()
        te_pred_pb = test_prediction[epoch_num].detach().numpy()
    else:
        tr_pred_pb = train_prediction[epoch_num].detach().numpy()
        te_pred_pb = test_prediction[epoch_num].detach().numpy()

    tr_pred = get_prediction(tr_pred_pb)
    te_pred = get_prediction(te_pred_pb)

    if train_s_t:
        tr_pred = tr_pred[train_s_t[0]:train_s_t[1]]
    if test_s_t:
        te_pred = te_pred[test_s_t[0]:test_s_t[1]]

    print("Model prediction length:")
    print(len(tr_pred), len(te_pred))

    if ofilter:
        label_tr_f = out_filter(label_tr, ofilter[0], ofilter[1])
        label_te_f = out_filter(label_te, ofilter[0], ofilter[1])

        tr_pred_f = out_filter(tr_pred, ofilter[0], ofilter[1])
        te_pred_f = out_filter(te_pred, ofilter[0], ofilter[1])
    print('=' * 50)
    print('=' * 50)
    print("Test Metrics:")
    print('=' * 50)
    print("No filter:")
    calculate_metrics(te_pred, label_te)
    if ofilter:
        print('=' * 50)
        print("Filter Pred")
        calculate_metrics(te_pred_f, label_te)
        print('=' * 50)
        print("Both filter")
        calculate_metrics(te_pred_f, label_te_f)
    print('=' * 50)
    print('=' * 50)
    print("train Metrics:")
    print('=' * 50)
    print("No filter:")
    calculate_metrics(tr_pred, label_tr)
    if ofilter:
        print('=' * 50)
        print("Filter Pred")
        calculate_metrics(tr_pred_f, label_tr)
        print('=' * 50)
        print("Both filter")
        calculate_metrics(tr_pred_f, label_tr_f)

    print('=' * 50)
    print('=' * 50)
    print("Test Visualization:")
    plt.figure(figsize=(10, 20))
    plt.subplot(411)
    plt.plot(agg_test)
    plt.title(f'agg_test {resident_id}')
    plt.subplot(412)
    plt.plot(agg_test * label_te)
    plt.title(f'EV work agg {resident_id}')
    plt.subplot(413)
    plt.plot(label_te)
    plt.title(f'Label Truth {resident_id}')
    plt.subplot(414)
    plt.plot(te_pred)
    plt.title(f'Label Prediction {resident_id}')
    plt.show()
    if ofilter:
        plt.figure(figsize=(10, 10))
        plt.subplot(211)
        plt.plot(label_te_f)
        plt.title(f'Label Truth filter {resident_id}')
        plt.subplot(212)
        plt.plot(te_pred_f)
        plt.title(f'Label Prediction filter {resident_id}')
        plt.show()
    print('=' * 50)
    print('=' * 50)
    print("Train Visualization:")
    plt.figure(figsize=(10, 20))
    plt.subplot(411)
    plt.plot(agg_train)
    plt.title(f'agg_train {resident_id}')
    plt.subplot(412)
    plt.plot(agg_train * label_tr)
    plt.title(f'EV work agg {resident_id}')
    plt.subplot(413)
    plt.plot(label_tr)
    plt.title(f'Label Truth {resident_id}')
    plt.subplot(414)
    plt.plot(tr_pred)
    plt.title(f'Label Prediction {resident_id}')
    plt.show()
    if ofilter:
        plt.figure(figsize=(10, 10))
        plt.subplot(211)
        plt.plot(label_tr_f)
        plt.title(f'Label Truth filter {resident_id}')
        plt.subplot(212)
        plt.plot(tr_pred_f)
        plt.title(f'Label Prediction filter {resident_id}')
        plt.show()
    if return_:
        return [te_pred, label_te, tr_pred, label_tr]


def get_transfer_model(input_model, reparam=False, cuda=True, verbose=False):
    """

    :param input_model:  model
    :param reparam:  boolean
    :param cuda: boolean
    :param verbose: boolean
    :return: model
    """
    model_ = copy.deepcopy(input_model)
    for name, param_ in model_.named_parameters():
        if name not in ['fc1.weight', 'fc1.bias',
                        'fc2.weight', 'fc2.bias',
                        'gn7.weight', 'gn7.bias',
                        'gn6.weight', 'gn6.bias',
                        'gn5.weight', 'gn5.bias',
                        'gn4.weight', 'gn4.bias',
                        'gn3.weight', 'gn3.bias',
                        'gn2.weight', 'gn2.bias',
                        'gn1.weight', 'gn1.bias']:
            param_.requires_grad = False
        else:
            param_.requires_grad = True

        if verbose:
            print(name, param_.requires_grad)
    if reparam:
        for m in model_.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    if cuda:
        model_ = model_.cuda()

    return model_


def predict_new_output(model, test_gen, seq_len, ofilter=None, return_=False, cuda=True):
    """

    :param model:  torch model
    :param test_gen:  generator
    :param seq_len:  int 20 or 10
    :param ofilter: list with two parameters [window, threshold]
    :param return_: boolean
    :param cuda:  boolean
    :return:  boolean
    """
    model.eval()
    with torch.no_grad():
        i = 0
        for data, label in test_gen:
            if i == 0:
                test_data_ = data
                test_label_ = label
                i += 1
            else:
                test_data_ = torch.cat([test_data_, data], dim=0)
                test_label_ = torch.cat([test_label_, label], dim=0)
        # Update and  fix the bug, here already aligned, no need to realigned.
        test_data_ = test_data_.type(torch.FloatTensor)
        test_label_ = test_label_.type(torch.FloatTensor).numpy()
        print(test_data_.shape, test_label_.shape)

        if cuda:
            test_data_ = test_data_.type(torch.FloatTensor).cuda()
            s = time.time()
            pred_ = model(test_data_)
            print('using time', time.time() - s)
            pred_ = pred_.detach().cpu().numpy()
            print("Model prediction output shape:", pred_.shape)
        else:
            s = time.time()
            pred_ = model(test_data_)
            print('using time', time.time() - s)
            pred_ = pred_.detach().numpy()

        pred_label = get_prediction(pred_)
        print('=' * 50)
        print("No filter:")
        calculate_metrics(pred_label, test_label_)

        if ofilter:
            label_f = out_filter(test_label_, ofilter[0], ofilter[1])
            pred_f = out_filter(pred_label, ofilter[0], ofilter[1])

            print('=' * 50)
            print("Filter Pred")
            calculate_metrics(pred_f, test_label_)
            print('=' * 50)
            print("Both filter")
            calculate_metrics(pred_f, label_f)

        if return_:
            return [pred_label, test_label_]


def get_parameters_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
