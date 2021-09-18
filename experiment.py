import os
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_default_tensor_type(torch.FloatTensor)
import time
from data_processing import TimeseriesDataset, get_resident_dt, data_scalar, train_test_data_split
from utils import *
from model.metrics import *
from model.loss import BinaryDiceLoss
import pickle
import warnings

def experiment(model, train_model, params, train_loader, test_loader):

    warnings.filterwarnings('ignore')
    print('\n')
    exec_t = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))
    print(f"Execute experiment: {exec_t}")
    if not os.path.exists(f"./checkpoint/{exec_t}"):
        os.mkdir(f"./checkpoint/{exec_t}")

    epoch= params['epoch']
    lr = params['lr']
    weight_decay =  params['weight_decay']
    use_cuda =  params['use_cuda']
    saved =params['saved']

    dice_loss = BinaryDiceLoss(smooth=1, p=2, reduction='mean')

    if use_cuda:
        model = model.cuda()
        dice_loss = BinaryDiceLoss(smooth=1, p=2, reduction='mean').cuda()

    optimizer = optim.Adam(model.parameters(),
                           lr=lr, betas=(0.9, 0.999),
                           weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.33, verbose=True,min_lr=0.00001, patience=125)

    f1 = F1Score()
    acc = Accuracy()
    recall = Recall()
    precision = Precision()

    # record list
    train_pred_record_dict = dict()

    train_epoch_loss_record = []
    train_epoch_acc_record = []
    train_epoch_recall_record = []
    train_epoch_precision_record = []
    train_epoch_f1_record = []

    test_pred_record_dict = dict()

    test_epoch_loss_record = []
    test_epoch_acc_record = []
    test_epoch_recall_record = []
    test_epoch_precision_record = []
    test_epoch_f1_record = []

    def train(epoch, model, loss_function, train_loader, use_cuda):
        model.train()

        train_pred_record_dict[epoch] = None

        train_loss_record = []
        train_label_record = []
        train_pred_record = []

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(torch.float32)
            target = target.to(torch.float32)
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)

            loss = loss_function(output, target)
            train_loss_record.append(loss.data.cpu().numpy())
            loss.backward()
            optimizer.step()
            train_label_record.append(target.cpu().detach().numpy())
            train_pred_record.append(output.cpu().detach().numpy())

        with torch.no_grad():

            train_pred = torch.tensor(np.vstack(np.array(train_pred_record)))

            train_label = torch.tensor(np.vstack(np.array(train_label_record)))

            train_acc = acc(train_pred, train_label).item()
            train_recall = recall(train_pred, train_label).item()
            train_precision = precision(train_pred, train_label).item()
            train_f1 = f1(train_pred, train_label).item()

            train_epoch_loss_record.append(np.mean((np.array(train_loss_record))))
            train_epoch_acc_record.append(train_acc)
            train_epoch_recall_record.append(train_recall)
            train_epoch_precision_record.append(train_precision)
            train_epoch_f1_record.append(train_f1)

            train_pred_record_dict[epoch] = train_pred

            print('Train:', epoch ,'Total training samples:', len(train_pred),'\n','loss:', train_epoch_loss_record[-1], 'acc:', '%.5f'%train_acc, 'recall',
                  '%.5f'%train_recall, 'precision', '%.5f'%train_precision, 'f1', '%.5f'%train_f1)
            if saved and epoch%5 == 0:
                torch.save(model,f"./checkpoint/{exec_t}/model_{epoch}_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))}.pth")


    def test(epoch, model, loss_function, test_loader, use_cuda):

        model.eval()

        test_pred_record_dict[epoch] = None

        test_loss_record = []
        test_label_record = []
        test_pred_record = []
        with torch.no_grad():

            for batch_idx, (data, target) in enumerate(test_loader):

                data = torch.tensor(data).to(torch.float32)
                target = torch.tensor(target).to(torch.float32)

                if use_cuda:
                    data, target = data.cuda(), target.cuda()

                output = model(data)

                loss = loss_function(output, target)

                test_label_record.append(target.data.cpu().numpy())
                test_pred_record.append(output.data.cpu().numpy())
                test_loss_record.append(loss.data.cpu().numpy())

            test_pred = torch.tensor(np.vstack(np.array(test_pred_record)))
            test_label = torch.tensor(np.vstack(np.array(test_label_record)))

            test_acc = acc(test_pred, test_label).item()
            test_recall = recall(test_pred, test_label).item()
            test_precision = precision(test_pred, test_label).item()
            test_f1 = f1(test_pred, test_label).item()

            test_epoch_loss_record.append(np.mean((np.array(test_loss_record))))
            test_epoch_acc_record.append(test_acc)
            test_epoch_recall_record.append(test_recall)
            test_epoch_precision_record.append(test_precision)
            test_epoch_f1_record.append(test_f1)

            test_pred_record_dict[epoch] = test_pred

            print('Test', epoch, 'Total testing samples:', len(test_pred),'\n','loss:', test_epoch_loss_record[-1],
                  'acc:', '%.5f'%test_acc, 'recall', '%.5f'%test_recall, 'precision', '%.5f'%test_precision, 'f1', '%.5f'%test_f1)
            if test_f1 >= max(test_epoch_f1_record):
                torch.save(model,f'./checkpoint/{exec_t}/model_{epoch}_f1_{test_f1}.pth')

    if train_model:
        for i in range(1, epoch+1):
            print(f"Epoch {i}:")
            train(i, model, dice_loss, train_loader, use_cuda=use_cuda)
            test(i, model, dice_loss, test_loader, use_cuda=use_cuda)
            print("\n")
    if not train_model:
        test(1, model, dice_loss, test_loader, use_cuda=use_cuda)

    train_record = dict()

    train_record['train_epoch_loss_record']= train_epoch_loss_record
    train_record['train_epoch_acc_record' ] = train_epoch_acc_record
    train_record['train_epoch_recall_record'] = train_epoch_recall_record
    train_record['train_epoch_precision_record'] = train_epoch_precision_record
    train_record['train_epoch_f1_record'] = train_epoch_f1_record


    test_record = dict()

    test_record['test_epoch_loss_record']= test_epoch_loss_record
    test_record['test_epoch_acc_record' ] = test_epoch_acc_record
    test_record['test_epoch_recall_record'] = test_epoch_recall_record
    test_record['test_epoch_precision_record'] = test_epoch_precision_record
    test_record['test_epoch_f1_record'] = test_epoch_f1_record


#     current_t = time.time()

    with open(f"./experiment_record/record_{exec_t}.pkl",'wb') as f:

        experiment_record = dict()
        experiment_record['train_record'] = train_record
        experiment_record['train_pred_record_dict'] = train_pred_record_dict
        experiment_record['test_record'] = test_record
        experiment_record['test_pred_record_dict'] = test_pred_record_dict

        pickle.dump(experiment_record,f)

        print(f"Saved experiment record in ./experiment_record/record_{exec_t}.pkl")

    return train_record, train_pred_record_dict, test_record,test_pred_record_dict



def experiment_lr(model, train_model, params, train_loader, test_loader):

    warnings.filterwarnings('ignore')
    print('\n')
    exec_t = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))
    print(f"Execute experiment: {exec_t}")
    if not os.path.exists(f"./checkpoint/{exec_t}"):
        os.mkdir(f"./checkpoint/{exec_t}")

    epoch= params['epoch']
    lr = params['lr']
    weight_decay =  params['weight_decay']
    use_cuda =  params['use_cuda']
    saved =params['saved']

    dice_loss = BinaryDiceLoss(smooth=1, p=2, reduction='mean')

    if use_cuda:
        model = model.cuda()
        dice_loss = BinaryDiceLoss(smooth=1, p=2, reduction='mean').cuda()

    optimizer = optim.Adam(model.parameters(),
                           lr=lr, betas=(0.9, 0.999),
                           weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.33, verbose=True,min_lr=0.00001, patience=125)

    f1 = F1Score()
    acc = Accuracy()
    recall = Recall()
    precision = Precision()

    # record list
    train_pred_record_dict = dict()

    train_epoch_loss_record = []
    train_epoch_acc_record = []
    train_epoch_recall_record = []
    train_epoch_precision_record = []
    train_epoch_f1_record = []

    test_pred_record_dict = dict()

    test_epoch_loss_record = []
    test_epoch_acc_record = []
    test_epoch_recall_record = []
    test_epoch_precision_record = []
    test_epoch_f1_record = []

    def train(epoch, model, loss_function, train_loader, use_cuda):
        model.train()

        train_pred_record_dict[epoch] = None

        train_loss_record = []
        train_label_record = []
        train_pred_record = []

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(torch.float32)
            target = target.to(torch.float32)
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)

            loss = loss_function(output, target)
            train_loss_record.append(loss.data.cpu().numpy())
            loss.backward()
            optimizer.step()
            scheduler.step(loss) # add in 0705
            train_label_record.append(target.cpu().detach().numpy())
            train_pred_record.append(output.cpu().detach().numpy())

        with torch.no_grad():

            train_pred = torch.tensor(np.vstack(np.array(train_pred_record)))

            train_label = torch.tensor(np.vstack(np.array(train_label_record)))

            train_acc = acc(train_pred, train_label).item()
            train_recall = recall(train_pred, train_label).item()
            train_precision = precision(train_pred, train_label).item()
            train_f1 = f1(train_pred, train_label).item()

            train_epoch_loss_record.append(np.mean((np.array(train_loss_record))))
            train_epoch_acc_record.append(train_acc)
            train_epoch_recall_record.append(train_recall)
            train_epoch_precision_record.append(train_precision)
            train_epoch_f1_record.append(train_f1)

            train_pred_record_dict[epoch] = train_pred

            print('Train:', epoch ,'Total training samples:', len(train_pred),'\n','loss:', train_epoch_loss_record[-1], 'acc:', '%.5f'%train_acc, 'recall',
                  '%.5f'%train_recall, 'precision', '%.5f'%train_precision, 'f1', '%.5f'%train_f1)
            if saved and epoch%5 == 0:
                torch.save(model,f"./checkpoint/{exec_t}/model_{epoch}_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))}.pth")


    def test(epoch, model, loss_function, test_loader, use_cuda):

        model.eval()

        test_pred_record_dict[epoch] = None

        test_loss_record = []
        test_label_record = []
        test_pred_record = []
        with torch.no_grad():

            for batch_idx, (data, target) in enumerate(test_loader):

                data = torch.tensor(data).to(torch.float32)
                target = torch.tensor(target).to(torch.float32)

                if use_cuda:
                    data, target = data.cuda(), target.cuda()

                output = model(data)

                loss = loss_function(output, target)

                test_label_record.append(target.data.cpu().numpy())
                test_pred_record.append(output.data.cpu().numpy())
                test_loss_record.append(loss.data.cpu().numpy())

            test_pred = torch.tensor(np.vstack(np.array(test_pred_record)))
            test_label = torch.tensor(np.vstack(np.array(test_label_record)))

            test_acc = acc(test_pred, test_label).item()
            test_recall = recall(test_pred, test_label).item()
            test_precision = precision(test_pred, test_label).item()
            test_f1 = f1(test_pred, test_label).item()

            test_epoch_loss_record.append(np.mean((np.array(test_loss_record))))
            test_epoch_acc_record.append(test_acc)
            test_epoch_recall_record.append(test_recall)
            test_epoch_precision_record.append(test_precision)
            test_epoch_f1_record.append(test_f1)

            test_pred_record_dict[epoch] = test_pred

            print('Test', epoch, 'Total testing samples:', len(test_pred),'\n','loss:', test_epoch_loss_record[-1],
                  'acc:', '%.5f'%test_acc, 'recall', '%.5f'%test_recall, 'precision', '%.5f'%test_precision, 'f1', '%.5f'%test_f1)
            if test_f1 >= max(test_epoch_f1_record):
                torch.save(model,f'./checkpoint/{exec_t}/model_{epoch}_f1_{test_f1}.pth')

    if train_model:
        for i in range(1, epoch+1):
            print(f"Epoch {i}:")
            train(i, model, dice_loss, train_loader, use_cuda=use_cuda)
            test(i, model, dice_loss, test_loader, use_cuda=use_cuda)
            print("\n")
    if not train_model:
        test(1, model, dice_loss, test_loader, use_cuda=use_cuda)

    train_record = dict()

    train_record['train_epoch_loss_record']= train_epoch_loss_record
    train_record['train_epoch_acc_record' ] = train_epoch_acc_record
    train_record['train_epoch_recall_record'] = train_epoch_recall_record
    train_record['train_epoch_precision_record'] = train_epoch_precision_record
    train_record['train_epoch_f1_record'] = train_epoch_f1_record


    test_record = dict()

    test_record['test_epoch_loss_record']= test_epoch_loss_record
    test_record['test_epoch_acc_record' ] = test_epoch_acc_record
    test_record['test_epoch_recall_record'] = test_epoch_recall_record
    test_record['test_epoch_precision_record'] = test_epoch_precision_record
    test_record['test_epoch_f1_record'] = test_epoch_f1_record


#     current_t = time.time()

    with open(f"./experiment_record/record_{exec_t}.pkl",'wb') as f:

        experiment_record = dict()
        experiment_record['train_record'] = train_record
        experiment_record['train_pred_record_dict'] = train_pred_record_dict
        experiment_record['test_record'] = test_record
        experiment_record['test_pred_record_dict'] = test_pred_record_dict

        pickle.dump(experiment_record,f)

        print(f"Saved experiment record in ./experiment_record/record_{exec_t}.pkl")

    return train_record, train_pred_record_dict, test_record,test_pred_record_dict

