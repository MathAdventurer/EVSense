#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# EVSense: Xudong Wang, Guoming Tang
# time:2021/5/29

import os
import datetime
import numpy as np
import pandas as pd
import torch
import sqlite3
import pickle
from utils import input_filter_df, Label_EV_data, str2time
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer


def Get_1min_data_Sqlite3(resident_id_list, site_name, datapath = '../datasets/raw_data',saved_path = None):
    """
    datapath:  strings datapath = './datasets/'
    resident_id_list: list int
    site_name: "Texas", "California", "NewYork" 
    
    """
    print("="*40)
    print("Start Query.")
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    dataids_str = ','.join([str(i) for i in resident_id_list])
    if site_name == "Texas":
        connection = sqlite3.connect(datapath +'1minute_data_austin.sqlite3')
        query = "SELECT * FROM '1minute_data_austin' WHERE dataid in ({})".format(dataids_str)
        if saved_path:
            saved_path = saved_path
        else:
            saved_path = datapath +'1minute_data_austin.csv'
    if site_name == "California":
        connection = sqlite3.connect(datapath +'1minute_data_california.sqlite3')
        query = "SELECT * FROM '1minute_data_california' WHERE dataid in ({})".format(dataids_str)
        if saved_path:
            saved_path = saved_path
        else:
            saved_path = datapath +'1minute_data_california.csv'
    if site_name == "NewYork":
        connection = sqlite3.connect(datapath +'1minute_data_newyork.sqlite3')
        query = "SELECT * FROM '1minute_data_newyork' WHERE dataid in ({})".format(dataids_str)
        if saved_path:
            saved_path = saved_path
        else:
            saved_path = datapath +'1minute_data_newyork.csv'
#     c = connection.cursor()
    print("="*40)
    print("Successfully Create the connection to database.")
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    print("="*40)
    print("Query data and transmit to Python Pandas")
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    df = pd.read_sql_query(query,connection)
    print("="*40)
    print("Finished Query data and transmit to Python Pandas")
    print(f"Total {df.shape[0]} record are found in DataBase.")
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
#     c.close()
    print("="*40) 
    print(f"Saved Query data in csv format at {saved_path}")
    df.to_csv(saved_path,index = False, encoding = 'utf-8')
    print(f"Saved Done!")
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    return df  


def Get_15min_data_Sqlite3(resident_id_list, site_name, datapath = '../datasets/raw_data/',saved_path = None):
    """
    datapath:  strings datapath = './datasets/' default
    resident_id_list: list int
    site_name: "Texas", "California", "NewYork" 
    
    """
    print("="*40)
    print("Start Query.")
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    dataids_str = ','.join([str(i) for i in resident_id_list])
    if site_name == "Texas":
        connection = sqlite3.connect(datapath +'15minute_data_austin.sqlite3')
        query = "SELECT * FROM '15minute_data_austin' WHERE dataid in ({})".format(dataids_str)
        if saved_path:
            saved_path = saved_path
        else:
            saved_path = datapath +'15minute_data_austin.csv'
    if site_name == "California":
        connection = sqlite3.connect(datapath +'15minute_data_california.sqlite3')
        query = "SELECT * FROM '15minute_data_california' WHERE dataid in ({})".format(dataids_str)
        if saved_path:
            saved_path = saved_path
        else:
            saved_path = datapath +'15minute_data_california.csv'
    if site_name == "NewYork":
        connection = sqlite3.connect(datapath +'15minute_data_newyork.sqlite3')
        query = "SELECT * FROM '15minute_data_newyork' WHERE dataid in ({})".format(dataids_str)
        if saved_path:
            saved_path = saved_path
        else:
            saved_path = datapath +'15minute_data_newyork.csv'
#     c = connection.cursor()
    print("="*40)
    print("Successfully Create the connection to database.")
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    print("="*40)
    print("Query data and transmit to Python Pandas")
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    df = pd.read_sql_query(query,connection)
    print("="*40)
    print("Finished Query data and transmit to Python Pandas")
    print(f"Total {df.shape[0]} record are found in DataBase.")
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
#     c.close()
    print("="*40) 
    print(f"Saved Query data in csv format at {saved_path}")
    df.to_csv(saved_path,index = False, encoding = 'utf-8')
    print(f"Saved Done!")
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    return df  



# Data Loader
class TimeseriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, seq_len=1):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len - 1)

    def __getitem__(self, index):
        return self.X[index:index + self.seq_len], self.y[index + self.seq_len - 1]


def get_resident_dt(dataset, id_num, time_period=None, input_filter=None, column_selection=None, raw_data = False, saved=False, check = False,check_path = None):
    """
    Using the dataset in pandas DataFrame format with index ['dataid','localminute']
    Please note that the value of dataid is integer
    Keep sure the localminute is in the datetime format
    Time_period is the time period dictionary with key start and end

    Update in 20210529: add raw data option to processing the raw csv read content
    Update in 20210605: drop solar and solar2 columns and set ev charing threshold temp is 1. Add option to get total df.
    Update in 20210607: Find some negative value in data grid, delete this 3 columns grid grid_l1 grid_l2.
    """
    pd.options.mode.chained_assignment = None  # default='warn'
    if check:
        if check_path:
            if time_period:
                if os.path.exists(check_path + f"{id_num}_{time_period['start']}_{time_period['end']}.csv"):
                    print(f"Found saved file \"{check_path}{id_num}_{time_period['start']}_{time_period['end']}.csv\" load from this file.")
                    temp = pd.read_csv(check_path +f"{id_num}_{time_period['start']}_{time_period['end']}.csv")
                    temp['localminute'] = temp['localminute'].apply(str2time)
                    temp.set_index(['localminute'], inplace=True)
                    temp.sort_index(inplace=True)
                    print(
                        f"Find the {id_num} in datasets with {temp.shape[0]} records of {time_period['start']} to {time_period['end']}.")
                    return temp
            if os.path.exists(check_path +f"{id_num}.csv"):
                print(f'Found saved file \"{check_path}{id_num}.csv\", load from this file.')
                temp = pd.read_csv(check_path +f"{id_num}.csv")
                temp['localminute'] = temp['localminute'].apply(str2time)
                temp.set_index(['localminute'], inplace=True)
                temp.sort_index(inplace=True)
                if time_period:
                    start = pd.Timestamp(datetime.datetime.strptime(time_period['start'], '%Y-%m-%d'))
                    end = pd.Timestamp(datetime.datetime.strptime(time_period['end'], '%Y-%m-%d'))
                    temp.sort_index(inplace=True)
                    temp = temp.loc[start:end]
                    print(
                        f"Find the {id_num} in datasets with {temp.shape[0]} records of {time_period['start']} to {time_period['end']}.")
                print(f'Find the {id_num} in datasets with {temp.shape[0]} records for {temp.shape[1]} colums.')
                return temp  
        else:
            if time_period:
                if os.path.exists(f"./{id_num}_{time_period['start']}_{time_period['end']}.csv"):
                    print(f"Found saved file \"./{id_num}_{time_period['start']}_{time_period['end']}.csv\" load from this file.")
                    temp = pd.read_csv(f"./{id_num}_{time_period['start']}_{time_period['end']}.csv")
                    temp['localminute'] = temp['localminute'].apply(str2time)
                    temp.set_index(['localminute'], inplace=True)
                    temp.sort_index(inplace=True)
                    print(
                        f"Find the {id_num} in datasets with {temp.shape[0]} records of {time_period['start']} to {time_period['end']}.")
                    return temp
            if os.path.exists(f"./{id_num}.csv"):
                print(f'Found saved file \"./{id_num}.csv\", load from this file.')
                temp = pd.read_csv(f"./{id_num}.csv")
                temp['localminute'] = temp['localminute'].apply(str2time)
                temp.set_index(['localminute'], inplace=True)
                temp.sort_index(inplace=True)
                if time_period:
                    start = pd.Timestamp(datetime.datetime.strptime(time_period['start'], '%Y-%m-%d'))
                    end = pd.Timestamp(datetime.datetime.strptime(time_period['end'], '%Y-%m-%d'))
                    temp.sort_index(inplace=True)
                    temp = temp.loc[start:end]
                    print(
                        f"Find the {id_num} in datasets with {temp.shape[0]} records of {time_period['start']} to {time_period['end']}.")
                print(f'Find the {id_num} in datasets with {temp.shape[0]} records for {temp.shape[1]} colums.')
                return temp
    if raw_data:
        dataset['localminute'] = dataset['localminute'].apply(str2time)
        dataset.set_index(['dataid', 'localminute'], inplace=True)
    temp = dataset.loc[id_num]
    
    if 'solar' in temp.columns:
        temp.drop(['solar'],axis = 1, inplace = True)
    if 'solar2' in temp.columns:
        temp.drop(['solar2'],axis = 1, inplace = True)
        
    if 'grid' in temp.columns:
        temp.drop(['grid'],axis = 1, inplace = True)
    if 'grid_l1' in temp.columns:
        temp.drop(['grid_l1'],axis = 1, inplace = True)     
    if 'grid_l2' in temp.columns:
        temp.drop(['grid_l2'],axis = 1, inplace = True)   
        
    print(f'Find the {id_num} in datasets with {temp.shape[0]} records for {temp.shape[1]} colums.')
    
    if time_period:
        start = pd.Timestamp(datetime.datetime.strptime(time_period['start'], '%Y-%m-%d'))
        end = pd.Timestamp(datetime.datetime.strptime(time_period['end'], '%Y-%m-%d'))
        temp.sort_index(inplace=True)
        temp = temp.loc[start:end]
        print(
            f"Find the {id_num} in datasets with {temp.shape[0]} records of {time_period['start']} to {time_period['end']}.")
    temp.sort_index(inplace=True)
    temp = temp.iloc[:, :-2]
    temp.loc[:, 'aggregate'] = temp.apply(lambda x: x.sum(), axis=1)
    if input_filter:
        temp.loc[:, 'aggregate'] = temp.apply(input_filter_df, args=(2,), axis=1)

#     temp.loc[:, 'label'] = temp.apply(Label_EV_data, args=(2,), axis=1)

    temp.loc[:, 'label'] = temp.apply(Label_EV_data, args=(1,), axis=1)
    
    selected_col = ['car1', 'car2', 'aggregate', 'label']
    if column_selection:
        if column_selection == 'total':
            selected_col = list(temp.columns)
        else:
            selected_col = ['car1', 'car2', 'aggregate', 'label'] + column_selection

    temp = temp.filter(selected_col)
    if saved:
        print("Start saving outcome.")
        if time_period:
            temp.to_csv(f"./{id_num}_{time_period['start']}_{time_period['end']}.csv", encoding='utf-8')
        else:
            temp.to_csv(f"./{id_num}.csv", encoding='utf-8')
        print("Start saving Done.")

    return temp

# 已经对齐时间
# example index = 0 seq_len = 20          return: x sequence 0:20 (20个) 0-19    y(19) 第20个状态



def data_scalar(data, method="Standard"):
    """
    data: support type DataFrame, Series, np.Array, list, torch.float32 Tensor
    method: "Standard","MinMax","MinMax"

    Update: 20210530 check the requires_grad

    """
    if method == "Standard":
        scalar_ = StandardScaler()
    if method == "MinMax":
        scalar_ = MinMaxScaler()
    if method == "Normal":
        scalar_ = Normalizer()
    if method not in ["Standard", "MinMax", "Normal"]:
        raise Exception(f"Not support method: {method}, Default is Standard, Please recheck input.")

    if type(data) is pd.core.frame.DataFrame:
        print("Input pandas Dataframe data, try to rescalar it.")
        if "aggregate" in data.columns:
            #             data['aggregate'] = pd.Series(scalar_.fit_transform(data['aggregate'].values))
            #             data['aggregate'] = pd.Series(scalar_.fit_transform(data['aggregate'].values.reshape(-1,1)).reshape(-1,)).reindex(data.index)
            #             data['aggregate']  = data['aggregate'].apply(lambda x: scalar_.fit_transform(x))
            data['aggregate'] = pd.DataFrame(
                scalar_.fit_transform(data['aggregate'].values.reshape(-1, 1)).reshape(-1, ), index=data.index)
        else:
            print(f"Error for ambiguous columns in Input DataFrame with shape {data.shape}")
    elif type(data) is pd.core.frame.Series:
        print("Input pandas Series data, try to rescalar it.")
        #         data = pd.Series(scalar_.fit_transform(data.values.reshape(-1,1)).reshape(-1,)).reindex(data.index)
        #         data  = data.apply(lambda x: scalar_.fit_transform(x))
        data = pd.DataFrame(scalar_.fit_transform(data.values.reshape(-1, 1)).reshape(-1, ), index=data.index).iloc[:,
               0]
    elif type(data) is np.ndarray:
        print("Input numpy array data, try to rescalar it.")
        data = scalar_.fit_transform(data.reshape(-1, 1)).reshape(-1, )
    elif type(data) is list:
        print("Input list data, try to transfer into numpy array and rescalar it.")
        data = scalar_.fit_transform(np.array(data).reshape(-1, 1)).reshape(-1, )
    elif type(data) is torch.Tensor:
        print("Input torch Tensor data, try to rescalar it.")
        with torch.no_grad():
            if data.requires_grad:
                data = torch.tensor(
                    scalar_.fit_transform(data.type(torch.FloatTensor).detach().numpy().reshape(-1, 1)).reshape(
                        -1, )).requires_grad_()
            else:
                data = torch.tensor(
                    scalar_.fit_transform(data.type(torch.FloatTensor).detach().numpy().reshape(-1, 1)).reshape(-1, ))
            assert data.dtype is torch.float32
    else:
        print(f"Input not support data type: {type(data)}, Please recheck input.")
    return data



def train_test_data_split(data, time_period, ratio=None):

    if time_period:

        if type(data) is pd.core.frame.DataFrame:

            train_start = pd.Timestamp(datetime.datetime.strptime(time_period['train']['start'], '%Y-%m-%d'))
            train_end = pd.Timestamp(datetime.datetime.strptime(time_period['train']['end'], '%Y-%m-%d'))
            test_start = pd.Timestamp(datetime.datetime.strptime(time_period['test']['start'], '%Y-%m-%d'))
            test_end = pd.Timestamp(datetime.datetime.strptime(time_period['test']['end'], '%Y-%m-%d'))

            train_data = data[train_start:train_end]
            test_data = data[test_start:test_end]

            agg_p_train = train_data['aggregate'].to_numpy()
            label_train = train_data['label'].to_numpy()
            agg_p_train = agg_p_train.reshape(len(agg_p_train), 1)
            label_train = label_train.reshape(len(label_train), 1)

            agg_p_test = test_data['aggregate'].to_numpy()
            label_test = test_data['label'].to_numpy()
            agg_p_test = agg_p_test.reshape(len(agg_p_test), 1)
            label_test = label_test.reshape(len(label_test), 1)

            return agg_p_train, label_train, agg_p_test, label_test

        else:

            print("Please input dataframe with time index to get certain period data.")

    if ratio:

        if type(data) is pd.core.frame.DataFrame:
            agg_p_train = data['aggregate'].to_numpy()[:int(ratio * data.shape[0])]
            label_train = data['label'].to_numpy()[:int(ratio * data.shape[0])]
            agg_p_train = agg_p_train.reshape(len(agg_p_train), 1)
            label_train = label_train.reshape(len(label_train), 1)

            agg_p_test = data['aggregate'].to_numpy()[int(ratio * data.shape[0]):]
            label_test = data['label'].to_numpy()[int(ratio * data.shape[0]):]
            agg_p_test = agg_p_test.reshape(len(agg_p_test), 1)
            label_test = label_test.reshape(len(label_test), 1)

            return agg_p_train, label_train, agg_p_test, label_test

        if type(data) is np.ndarray:
            train_data = data[:int(ratio * data.shape[0])]
            test_data = data[int(ratio * data.shape[0]):]

            return train_data, test_data


def Synthesis_data(resident_, EV_sessions, start, end, option=[3, [2880, 5760], 5]):
    """
    resident_: DataFrame with index is datatime format, with column: 'aggregate', 'car1', 'label', no charging events
    EV_sessions: EV charging profile from Pecanstreet, dictionary, with key is from 1 to 6 corresponding to residents.
    start: string format, eg: '2014-05-01'
    end: string format, eg: '2014-07-31'
    option: list [interval to calculate the roughly embedding events,
    [between two charging events min, between two charging events max],
    compensation events number to ensure inserts until the end]
    """
    interval_cal = option[0]
    interval_in_session = option[1]
    resident_df = resident_.copy(deep=True)[start:end]
    series_len = len(resident_df)
    curent_index = 0
    compensation = option[2]
    ####
    d_1 = np.random.randint(1, 7)
    print('Select EV domain:', d_1)
    session_num = int(np.ceil(series_len / (60 * 24 * interval_cal)) + compensation)
    session_id = list(np.random.choice(a=range(len(EV_sessions[d_1])), size=session_num, replace=False, p=None))
    session_len = [len(EV_sessions[d_1].iloc[id_]) for id_ in session_id]
    start_point = int(np.random.choice(range(720, 1440), 1))
    curent_index += start_point
    print('session_id: ', session_id)
    print('session_len: ', session_len)
    print('start insert: ', curent_index)
    for i in range(session_num):
        if int(curent_index + session_len[i]) <= int(series_len - 1):
            resident_df['car1'].iloc[int(curent_index):int(curent_index + session_len[i]), ] = EV_sessions[d_1].iloc[
                session_id[i]].values
            resident_df['label'].iloc[int(curent_index):int(curent_index + session_len[i]), ] = 1.0
            curent_index = int(curent_index + session_len[i])

            interval_ = int(np.random.choice(range(interval_in_session[0], interval_in_session[1]), 1))

            print(f'finish insert {session_id[i]}, next interval is {interval_}')

            resident_df['label'].plot(figsize=(20, 3))
            plt.show()
            resident_df['car1'].plot(figsize=(20, 3))
            plt.show()
            if int(curent_index + interval_) > int(series_len - 1) and i != session_num - 1:
                print('Stop fill done!')
                break
            else:
                curent_index = int(curent_index + interval_)
                print('continue insert: ', curent_index)
        else:
            break
    # generate the aggregate power with the inserted EV charging profile.
    resident_df['aggregate'] = resident_df['aggregate'] + resident_df['car1']
    return resident_df
