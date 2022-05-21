#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# time:2021/6/16
# EVSense: Xudong Wang, Guoming Tang

# Start this file in 20210615
# Update in 20210618, adjust the figure size.

import os
import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def EV_session_analysis(df, resident_id, time_period, saved_path, verbose=False, label_bins_num=3, label_num_std=10):
    """
    Start Package this analysis in 20210616
    Input the df with the index is the localminute(in Pandas DataFrame.datetime format)
    Auto-saved the EV-sessons analysis at the target dictionary path.
    df: Input data Frame
    saved_path: string, path
    return the EV session analysis dataframe and relabel the EV charging event.

    """
    print("=" * 50)
    print(f"EV Charging sessions analysis for resident {resident_id}")
    # figure 1 EV_session_meter_histogram
    plt.figure()
    n, bins, patches = plt.hist(df['car1'], bins="doane")  # default is bins="auto"
    plt.title(f"resident {resident_id} EV Meter records (per min) Histogram")
    plt.xlabel("Power: Kwh")
    plt.ylabel('Frequency')
    plt.savefig(saved_path + f'{resident_id}_EV_session_meter_histogram.pdf', format='pdf')
    if verbose:
        plt.show()
    plt.close()

    # Analysis the distribution and relabel it.
    diff = np.diff(bins)[0]
    print('The value interval for histgram bins is', diff)
    print(f"Current histogram bins for determinant EV charging session label is {label_bins_num}")
    # Using the bin to relabel the EV charging sessions.
    top_bins_idx = np.argsort(n)[-label_bins_num:][::-1]
    thresh = np.sort([bins[top_bins_idx[0]], bins[top_bins_idx[1]], bins[top_bins_idx[2]]])
    thresh_0, thresh_1, thresh_2 = thresh

    # pts_0 = df['car1'][(df['car1'] >= thresh_0) & (df['car1'] <= (thresh_0 + diff))] #Bottom bin
    pts_0 = df['car1'][
        (df['car1'] >= thresh_1) & (df['car1'] <= (thresh_1 + 2 * diff))]  # Middle bin + another
    pts_1 = df['car1'][
        (df['car1'] >= thresh_2) & (df['car1'] <= (thresh_2 + diff))]  # Top bin

    mean_0 = np.mean(pts_0)
    mean_1 = np.mean(pts_1)
    std_0 = np.std(pts_0)
    std_1 = np.std(pts_1)

    print(f"OFF: mean:{mean_0} Kw, std:{std_0}")
    print(f"ON: mean:{mean_1} Kw, std:{std_1}")
    print(f"Using {label_bins_num} bins threshold:", thresh_0, thresh_1, thresh_2)
    print(f"Current {label_num_std} times std of the OFF state (Upper bound) to determine the EV charging sessions:",
          mean_0 + label_num_std * std_0)

    # relabel the car label
    df['label'] = np.ones(len(df))
    df['label'].loc[(df['car1'] <= (mean_0 + label_num_std * std_0))] = 0

    # figure 2+ EV_session_visualization
    if time_period:
        start = pd.Timestamp(datetime.datetime.strptime(time_period['start'], '%Y-%m-%d'))
        end = pd.Timestamp(datetime.datetime.strptime(time_period['end'], '%Y-%m-%d'))
        df.sort_index(inplace=True)
        temp = df.loc[start:end]

        fig, ax1 = plt.subplots(figsize=(15, 5))
        ax2 = ax1.twinx()
        ax1.plot(temp['label'], 'r-')
        ax2.plot(temp['car1'], 'b--')
        plt.title(f'resident: {resident_id} EV charging sessions visualization')
        ax1.set_ylabel('Label for EV Charging Event', color='r')
        ax2.set_ylabel('Car', color='b')
        plt.savefig(saved_path + f'{resident_id}_EV_charging_sessions_visualization.pdf', format='pdf')
        if verbose:
            plt.show()
        plt.close()

        plt.figure(figsize=(14, 7))
        df['car1'].loc['2018-03-20':'2018-03-21'].plot(figsize=(14, 7), style=['-'])
        (df['car1'] * df['label']).loc['2018-03-20':'2018-03-21'].plot(figsize=(14, 7), style=['ro'])
        # plt.title(f"EV charging session for resident {resident_id} with label (Red dots)")
        plt.ylabel('Kw')
        plt.savefig(saved_path + f'{resident_id}_EV_charging_sessions_visualization_demo.pdf', format='pdf')
        if verbose:
            plt.show()
        plt.close()
    else:
        fig, ax1 = plt.subplots(figsize=(15, 5))
        ax2 = ax1.twinx()
        ax1.plot(df['label'], 'r-')
        ax2.plot(df['car1'], 'b--')
        plt.title(f'resident: {resident_id} EV Charging sessions visualization')
        ax1.set_ylabel('Label for EV Charging Event', color='r')
        ax2.set_ylabel('Car', color='b')
        plt.savefig(saved_path + f'{resident_id}_EV_Charging_sessions_visualization.pdf', format='pdf')
        if verbose:
            plt.show()
        plt.close()

    # figure 3 EV Charging_agg_vs_car curve
    plt.figure(figsize=(18, 5))
    df.filter(['aggregate', 'car1']).plot()
    plt.title(f'resident {resident_id} Aggregate Power Vs EV charging Power')
    plt.ylabel('Kw')
    plt.savefig(saved_path + f'{resident_id}_EV_charging_agg_vs_car.pdf', format='pdf')
    if verbose:
        plt.show()
    plt.close()

    # Analysis the Detail EV session and return the EV session
    print("=" * 50)
    ses_df = get_EV_charging_session(df, resident_id, saved_path, verbose, duration_fig_num=15)

    # Analysis the start and end charging session time.

    plt.figure(figsize=(6, 6))
    ax = ses_df['startTime'].dt.tz_localize('UTC').dt.tz_convert("America/Chicago").dt.hour.plot.hist(bins=24,
                                                                                                      title=f"Histogram of startTime Hour of Day for resident {resident_id}")
    ax.set_xlabel("Hour of Day")
    plt.savefig(saved_path + f'{resident_id}_EV_charging_startTime_Histogram.pdf', format='pdf')
    if verbose:
        plt.show()
    plt.close()

    plt.figure(figsize=(6, 6))
    ax = ses_df['endTime'].dt.tz_localize('UTC').dt.tz_convert("America/Chicago").dt.hour.plot.hist(bins=24,
                                                                                                    title=f"Histogram of endTime Hour of Day for resident {resident_id}")
    ax.set_xlabel("Hour of Day")
    plt.savefig(saved_path + f'{resident_id}_EV_charging_endTime_Histogram.pdf', format='pdf')
    if verbose:
        plt.show()
    plt.close()

    # Start vs End time
    from sklearn.cluster import DBSCAN
    X = [ses_df['startTime'].dt.tz_localize('UTC').dt.tz_convert("America/Chicago").dt.hour + ses_df[
        'startTime'].dt.minute * 1 / 60,
         ses_df['endTime'].dt.tz_localize('UTC').dt.tz_convert("America/Chicago").dt.hour + ses_df[
             'endTime'].dt.minute * 1 / 60]
    X = np.array(X).T
    model = DBSCAN(eps=2, min_samples=int(np.round(0.1 * X.shape[0])), random_state=0)
    y = model.fit_predict(X)
    # print(y)
    plt.figure(figsize=(6, 6))
    COLOR = ['green', 'red', 'blue', 'black', 'gray']
    for idx in range(len(y)):
        plt.scatter(X[idx, 0], X[idx, 1], color=COLOR[y[idx]])
        plt.xlabel("startTime Hour of Day")
        plt.ylabel("endTime Hour of Day")
        plt.title(f"startTime vs. endTime (Hour of Day) for resident {resident_id}")
    plt.plot([0, 24], [0, 24], 'k-')
    plt.savefig(saved_path + f'{resident_id}_EV_charging_start_vs_endTime.pdf', format='pdf')
    if verbose:
        plt.show()
    plt.close()

    # Histogram of energy per charge

    plt.figure(figsize=(6, 6))
    ax = ses_df['energy'].plot.hist(bins=24, title=f"Histogram of Energy per Charge for resident {resident_id}")
    ax.set_xlabel("Energy per Charge (kWh)")
    plt.savefig(saved_path + f'{resident_id}_EV_charging_energy_percharge.pdf', format='pdf')
    if verbose:
        plt.show()
    plt.close()

    # Histogram of the energy for month and season
    plt.figure(figsize=(6, 6))
    monthCounter = [sum(ses_df['startTime'].dt.tz_localize('UTC').dt.tz_convert("America/Chicago").dt.month == m) for m
                    in range(1, 13)]
    plt.bar(range(1, 13), monthCounter)
    plt.xlabel('Month')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Energy monthly Charge for resident {resident_id}')
    plt.savefig(saved_path + f'{resident_id}_EV_charging_frequency_monthly.pdf', format='pdf')
    if verbose:
        plt.show()
    plt.close()

    plt.figure(figsize=(6, 6))
    seasonCounter = [sum(monthCounter[0:3]), sum(monthCounter[3:6]), sum(monthCounter[6:9]), sum(monthCounter[9:])]
    plt.bar(["Winter", "Spring", "Summer", "Autumn"], seasonCounter)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Energy seasonal Charge for resident {resident_id}')
    plt.savefig(saved_path + f'{resident_id}_EV_charging_frequency_seasonal.pdf', format='pdf')
    if verbose:
        plt.show()
    plt.close()

    plt.figure(figsize=(6, 6))
    dayOfWeekCounter = [
        sum(ses_df['startTime'].dt.tz_localize('UTC').dt.tz_convert("America/Chicago").dt.dayofweek == dow) / 52 for dow
        in range(0, 7)]
    plt.bar(["M", "Tu", "W", "Th", "F", "Sa", "Su"], dayOfWeekCounter)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Energy week Charge for resident {resident_id}')
    plt.savefig(saved_path + f'{resident_id}_EV_charging_frequency_weekly.pdf', format='pdf')
    if verbose:
        plt.show()
    plt.close()

    plt.figure(figsize=(14, 7))
    ses_df.boxplot(column='energy', by='startHour', figsize=(14, 7))
    plt.savefig(saved_path + f'{resident_id}_EV_charging_energybystartHour.pdf', format='pdf')
    if verbose:
        plt.show()
    plt.close()

    return ses_df


def get_EV_charging_session(df, resident_id, saved_path, verbose=False, duration_fig_num=15):
    ses_df = pd.DataFrame(columns=['power', 'startTime', 'endTime',
                                   'duration', 'peakPower', 'tailDuration', 'energy'])
    sessions = []
    startTime = []
    endTime = []
    duration = []
    start_idx = df.index[0]
    end_idx = df.index[0]
    prev_state = df['label'][0]
    for idx, row in df.iterrows():
        state = row['label']
        if state != prev_state:
            if state == 1:
                start_idx = idx - pd.Timedelta('1 min')
                df['label'].loc[start_idx] = 1
            else:
                end_idx = idx
                sessions.append(df['car1'].loc[start_idx:end_idx])
                startTime.append(start_idx)
                endTime.append(end_idx)
                duration.append(end_idx - start_idx)
                start_idx = end_idx
        prev_state = state
    ses_df['power'] = sessions
    ses_df['startTime'] = startTime
    ses_df['endTime'] = endTime
    ses_df['duration'] = duration

    for idx, row in ses_df.iterrows():
        ses_df['peakPower'].loc[idx] = np.max(row['power'])
        ses_df['energy'].loc[idx] = np.trapz(row['power'],
                                             row['power'].index.astype(np.int64) / 10 ** 9) / 3.6e3  # [kWh]

    ses_df_orig = ses_df.copy()
    print("Plot the preliminary figure of the EV charging sessions.")
    plt.figure()
    ses_df['duration'].dt.total_seconds().div(60).astype(int).plot.hist(bins=50, title=f"{resident_id}: duration")
    plt.savefig(saved_path + f'{resident_id}_EV_charging_sessions_duration_origin.pdf', format='pdf')
    if verbose:
        plt.show()
    plt.close()

    plt.figure()
    ses_df['energy'].plot.hist(bins=50, title=f"{resident_id}: energy")
    plt.savefig(saved_path + f'{resident_id}_EV_charging_sessions_energy_origin.pdf', format='pdf')
    if verbose:
        plt.show()
    plt.close()

    plt.figure()
    ses_df['peakPower'].plot.hist(bins=50, title=f"{resident_id}: peakPower")
    plt.savefig(saved_path + f'{resident_id}_EV_charging_sessions_peakPower_origin.pdf', format='pdf')
    if verbose:
        plt.show()
    plt.close()

    #  Remove some outliers session.
    print("Plot the figures of the EV charging sessions after remove outliers.")
    ses_df = ses_df_orig.copy()
    prevLen = len(ses_df_orig)
    ses_df = ses_df[ses_df['peakPower'] > 1.2]
    print("{} sessions removed ({}%), {} remaining".format(prevLen - len(ses_df.index),
                                                           np.round((prevLen - len(ses_df.index)) / prevLen, 2),
                                                           len(ses_df.index)))

    prevLen = len(ses_df)
    ses_df = ses_df[ses_df['energy'] >= 0.1]
    print("{} sessions removed ({}%), {} remaining".format(prevLen - len(ses_df.index),
                                                           np.round((prevLen - len(ses_df.index)) / prevLen, 2),
                                                           len(ses_df.index)))

    prevLen = len(ses_df)
    ses_df = ses_df[ses_df['duration'].dt.total_seconds() >= 120]
    print("{} sessions removed ({}%), {} remaining".format(prevLen - len(ses_df.index),
                                                           np.round((prevLen - len(ses_df.index)) / prevLen, 2),
                                                           len(ses_df.index)))

    plt.figure()
    ses_df['duration'].dt.total_seconds().div(60).astype(int).plot.hist(bins=50, title=f"{resident_id}: duration")
    plt.savefig(saved_path + f'{resident_id}_EV_charging_sessions_duration_cleaned.pdf', format='pdf')
    if verbose:
        plt.show()
    plt.close()

    plt.figure()
    ses_df['energy'].plot.hist(bins=50, title=f"{resident_id}: energy")
    plt.savefig(saved_path + f'{resident_id}_EV_charging_sessions_energy_cleaned.pdf', format='pdf')
    if verbose:
        plt.show()
    plt.close()

    plt.figure()
    ses_df['peakPower'].plot.hist(bins=50, title=f"{resident_id}: peakPower")
    plt.savefig(saved_path + f'{resident_id}_EV_charging_sessions_peakPower_cleaned.pdf', format='pdf')
    if verbose:
        plt.show()
    plt.close()

    # Analysis the EV charging session.

    window = 5
    ses_df['tailBin'] = None
    for idx, row in ses_df.iterrows():
        convolve = np.convolve(row['power'], np.ones(window) * 1 / window, mode='valid')
        diff = np.diff(convolve)
        thresh_idxs = np.where(diff < -1e-1)[0]
        thresh_idxs = thresh_idxs[np.where(thresh_idxs > np.round(len(row['power']) * 0.1))[0]]
        if len(thresh_idxs) == 0:
            print(diff)
            tailStart = int(np.round(len(row['power']) * 0.9))
        else:
            tailStart = thresh_idxs[0]
        tailDuration = len(row['power']) - tailStart
        ses_df['tailDuration'].loc[idx] = tailDuration
        try:
            tail = row['power'].values[tailStart:]
            numBins = 5
            binWidth = tail[0] / numBins
            tailBin = [0]
            for B in reversed(range(1, numBins)):
                tailBin.append(np.where(tail >= B * binWidth)[0][-1] / tailDuration)
            tailBin.append(1)
            ses_df['tailBin'].loc[idx] = tailBin
        except IndexError as e:
            print("IndexError Details : " + str(e))
            pass
        continue

    # Plot the EV Charging sessions tailDuration
    plt.figure()
    ses_df['tailDuration'].plot.hist(bins=50)
    plt.title(f"{resident_id}: Charging session's tailDuration")
    plt.savefig(saved_path + f'{resident_id}_EV_charging_sessions_tailDuration.pdf', format='pdf')
    if verbose:
        plt.show()
    plt.close()

    plt.figure()
    ses_df['tailDuration'][ses_df['tailDuration'] < 150].plot.hist(bins=50)
    plt.title(f"{resident_id}: Charging session's tailDuration")
    plt.savefig(saved_path + f'{resident_id}_EV_charging_sessions_tailDuration_cleaned.pdf', format='pdf')
    if verbose:
        plt.show()
    plt.close()

    # plot the tail duration for the EV

    lim = duration_fig_num
    for idx, row in ses_df[ses_df['tailDuration'] < 8].iterrows():
        if lim <= 0:
            break
        try:
            tail = row['power'][len(row['power']) - int(row['tailDuration']):]
            plt.figure()
            ses_df['power'].loc[idx].plot(
                title="duration={}, tailDuration={}".format(row['duration'], row['tailDuration']))
            tail.plot(style='r.-')
            plt.savefig(saved_path + f'{resident_id}_EV_charging_sessions_tailDuration_Short_duration_{lim}.pdf',
                        format='pdf')
            if verbose:
                plt.show()
            plt.close()
        except IndexError as e:
            print("IndexError Details : " + str(e))
            pass
        lim -= 1
        continue

    lim = duration_fig_num
    for idx, row in ses_df[ses_df['tailDuration'] >= 8].iterrows():
        if lim <= 0:
            break
        try:
            tail = row['power'][len(row['power']) - int(row['tailDuration']):]
            plt.figure()
            ses_df['power'].loc[idx].plot(
                title="duration={}, tailDuration={}".format(row['duration'], row['tailDuration']))
            tail.plot(style='r.-')
            plt.savefig(saved_path + f'{resident_id}_EV_charging_sessions_tailDuration_Long_duration_{lim}.pdf',
                        format='pdf')
            if verbose:
                plt.show()
            plt.close()
        except IndexError as e:
            print("IndexError Details : " + str(e))
            pass
        lim -= 1
        continue

    ses_df['startHour'] = ses_df['startTime'].dt.tz_localize('UTC').dt.tz_convert("America/Chicago").dt.hour
    ses_df['endHour'] = ses_df['endTime'].dt.tz_localize('UTC').dt.tz_convert("America/Chicago").dt.hour

    ses_df.to_csv(saved_path + f'{resident_id}_EV_charging_sessions_analysis.csv', encoding='utf-8', index=False)
    #     print("Saved Done!")
    return ses_df
