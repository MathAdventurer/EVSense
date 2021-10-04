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
from model.loss import BinaryDiceLoss, CORALLOSS, CORAL
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


def transfer_learning(model, train_model, params, train_loader, source_loader, test_loader = None):
    
    warnings.filterwarnings('ignore') #Filter warning, need buid-in package warnings
    
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
    transfer_loss_weight = params['transfer_loss_weight']

    dice_loss = BinaryDiceLoss(smooth=1, p=2, reduction='mean')

    if use_cuda:
        model = model.cuda()
        dice_loss = BinaryDiceLoss(smooth=1, p=2, reduction='mean').cuda()

    optimizer = optim.Adam(model.parameters(),
                           lr=lr, betas=(0.9, 0.999),
                           weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.33, verbose=True,min_lr=0.00001, patience=125)

    # Metrics setting
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
    train_epoch_transfer_loss = []
    train_epoch_total_loss = []
    
    test_pred_record_dict = dict()

    test_epoch_loss_record = []
    test_epoch_acc_record = []
    test_epoch_recall_record = []
    test_epoch_precision_record = []
    test_epoch_f1_record = []

    
    def transfer_train(epoch, model, loss_function, train_loader, source_loader, use_cuda):
        
        model.train()

        train_pred_record_dict[epoch] = None

        train_loss_record = []
        train_label_record = []
        train_pred_record = []

        for pahse in ['src','tar']:
            
            if pahse == 'src': 

                optimizer.zero_grad()

                temp_src_loss = torch.tensor(0.0).to(torch.float32).cuda() 

                temp_src = None #use to cache the training data

                for batch_idx, (src_data, src_target) in enumerate(source_loader):

                    src_data = src_data.to(torch.float32)
                    src_target = src_target.to(torch.float32)
                    if use_cuda:
                        src_data, src_target = src_data.cuda(), src_target.cuda()

                    if temp_src is None:
                        temp_src = src_data.clone().detach().to(torch.float32)
                        if use_cuda:
                            temp_src = temp_src.cuda()
                    else:
                        temp_src = torch.vstack((temp_src,src_data))

                    output = model(src_data)

                    loss_1 = loss_function(output, src_target)

                    temp_src_loss += loss_1

                    train_loss_record.append(loss_1.data.cpu().numpy())
                    train_label_record.append(src_target.cpu().detach().numpy())
                    train_pred_record.append(output.cpu().detach().numpy())

            else:

                temp_tar = None

                for batch_idx, (tar_data, tar_target) in enumerate(train_loader):

                    tar_data = tar_data.to(torch.float32)

                    tar_target = tar_target.to(torch.float32)

                    if use_cuda:
                        tar_data, tar_target = tar_data.cuda(), tar_target.cuda()

                    if temp_tar is None:
                        temp_tar = tar_data.to(torch.float32)
                        if use_cuda:
                            temp_tar = temp_tar.cuda()
                    else:
                        temp_tar = torch.vstack((temp_tar,tar_data))
                        

        domain_loss = model(temp_tar,transfer = True, x_source = temp_src, domain_loss = CORAL)
        loss = temp_src_loss + transfer_loss_weight*domain_loss                          
        loss.backward()
        optimizer.step()
        scheduler.step(loss) # Step the learning rate scheduler
        
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
            
            train_epoch_transfer_loss.append((transfer_loss_weight*domain_loss).data.cpu().numpy())
            train_epoch_total_loss.append(loss.data.cpu().numpy())
            
            print('Transfer Train:', epoch ,'Total source samples:', temp_src.shape[0],'Total target samples:', temp_tar.shape[0],'\n',
                  'Supervised loss of source data:', '%.5f'%train_epoch_loss_record[-1], 'acc:', '%.5f'%train_acc, 'recall',
                      '%.5f'%train_recall, 'precision', '%.5f'%train_precision, 'f1', '%.5f'%train_f1,'\n',
                 'CORAL loss of target domain data:', '%.5f'%(transfer_loss_weight*domain_loss),f'with weight{transfer_loss_weight}','\n',
                 'Total loss:','%.5f'%loss.data.cpu().numpy())
            
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
            transfer_train(i, model, dice_loss, train_loader, source_loader, use_cuda = use_cuda)
            if test_loader:
                test(i, model, dice_loss, test_loader, use_cuda=use_cuda)
                print("\n")
    if not train_model:
        if test_loader:
            test(1, model, dice_loss, test_loader, use_cuda=use_cuda)
        else:
            print("Please give the target domain test data.")
                

    train_record = dict()

    train_record['train_epoch_loss_record']= train_epoch_loss_record
    train_record['train_epoch_acc_record' ] = train_epoch_acc_record
    train_record['train_epoch_recall_record'] = train_epoch_recall_record
    train_record['train_epoch_precision_record'] = train_epoch_precision_record
    train_record['train_epoch_f1_record'] = train_epoch_f1_record
    
    train_record['train_epoch_transfer_loss'] = train_epoch_transfer_loss
    train_record['train_epoch_total_loss'] = train_epoch_total_loss


    test_record = dict()

    test_record['test_epoch_loss_record']= test_epoch_loss_record
    test_record['test_epoch_acc_record' ] = test_epoch_acc_record
    test_record['test_epoch_recall_record'] = test_epoch_recall_record
    test_record['test_epoch_precision_record'] = test_epoch_precision_record
    test_record['test_epoch_f1_record'] = test_epoch_f1_record



    with open(f"./experiment_record/record_{exec_t}.pkl",'wb') as f:

        experiment_record = dict()
        experiment_record['train_record'] = train_record
        experiment_record['train_pred_record_dict'] = train_pred_record_dict
        experiment_record['test_record'] = test_record
        experiment_record['test_pred_record_dict'] = test_pred_record_dict

        pickle.dump(experiment_record,f)

        print(f"Saved experiment record in ./experiment_record/record_{exec_t}.pkl")

    return train_record, train_pred_record_dict, test_record,test_pred_record_dict