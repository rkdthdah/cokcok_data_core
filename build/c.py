import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import math
import os
import itertools
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def getData(folder_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, folder_name)

    data_list = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            data_list[file_name[:-4]] = readFile(folder_path + file_name)
    
    return data_list

def readFile(directory):
    df = pd.read_csv(directory)

    # 라디안 -> 도 변환
    for i in range(len(df)):
        df.at[i, 'RRX'] = df.at[i, 'RRX'] * 180 / math.pi
        df.at[i, 'RRY'] = df.at[i, 'RRY'] * 180 / math.pi
        df.at[i, 'RRZ'] = df.at[i, 'RRZ'] * 180 / math.pi
    
    return df

def findPeak(df, window_size, acc_threshold, rr_threshold, feat_name, feat_threshold, peak_threshold):
    # 피크 찾기
    # peak_range 만들기 - 해당 index에서 몇개의 peak 범위가 겹치는가 - 같은 특징에서 중복되는 peak 범위는 제외
    peak_range = [0] * len(df)

    acc_peaks = {}

    acc_peaks['ax_peaks_plus'], _ = find_peaks(df['AccelerationX'], height=acc_threshold)
    acc_peaks['ay_peaks_plus'], _ = find_peaks(df['AccelerationY'], height=acc_threshold)
    acc_peaks['az_peaks_plus'], _ = find_peaks(df['AccelerationZ'], height=acc_threshold)

    acc_peaks['ax_peaks_minus'], _ = find_peaks(-df['AccelerationX'], height=acc_threshold)
    acc_peaks['ay_peaks_minus'], _ = find_peaks(-df['AccelerationY'], height=acc_threshold)
    acc_peaks['az_peaks_minus'], _ = find_peaks(-df['AccelerationZ'], height=acc_threshold)

    for feat_peak in acc_peaks.values():
        for peak in feat_peak:
            for index in range(int(peak - window_size/2), int(peak + window_size/2)):
                if index >= len(df) - 1: continue

                peak_range[index] += 1


    rot_peaks = {}

    rot_peaks['rx_peaks_plus'], _ = find_peaks(df['RRX'], height=rr_threshold)
    rot_peaks['ry_peaks_plus'], _ = find_peaks(df['RRY'], height=rr_threshold)
    rot_peaks['rz_peaks_plus'], _ = find_peaks(df['RRZ'], height=rr_threshold)

    rot_peaks['rx_peaks_minus'], _ = find_peaks(-df['RRX'], height=rr_threshold)
    rot_peaks['ry_peaks_minus'], _ = find_peaks(-df['RRY'], height=rr_threshold)
    rot_peaks['rz_peaks_minus'], _ = find_peaks(-df['RRZ'], height=rr_threshold)

    for feat_peak in rot_peaks.values():
        last_range = -window_size
        for peak in feat_peak:
            index = 0
            for index in range(int(peak - window_size/2), int(peak + window_size/2)):
                if index >= len(df) or index <= last_range: continue

                peak_range[index] += 1
            
            last_range = index

    # 피크 찾기 - peak_threshold보다 peak_range가 많이 겹치는 구간에서
    # feat_name이 feat_threshold보다 높게 튀는 곳 중 가장 큰 값(절대값)을 가진 곳이 peak.
    # 찾은 peak 인덱스를 res_peak에 저장
    res_peaks = []
    last_start = 0
    last_max = 0
    for index in range(int(window_size / 2), len(peak_range) - int(window_size / 2)):

        if peak_range[index] >= peak_threshold and peak_range[index - 1] < peak_threshold:
            if last_start + window_size < index:
                last_start = index - int(window_size / 2)
        
        if peak_range[index] < peak_threshold and peak_range[index - 1] >= peak_threshold:
            end = index + int(window_size / 2)

            if index - last_start > window_size * 3:

                accx_peaks_1, _ = find_peaks(-df.loc[last_start:int((end + last_start)/2), feat_name], height=feat_threshold)
                max_peak = 0
                max_index = 0

                for i in accx_peaks_1:
                    if abs(df.at[last_start + i, feat_name]) > max_peak:
                        max_peak = abs(df.at[last_start + i, feat_name])
                        max_index = last_start + i
                if max_index != 0:
                    if res_peaks and max_index - res_peaks[-1] <= int(window_size * 1.5):
                        if max_peak > last_max:
                            res_peaks.pop()
                            res_peaks.append(max_index)
                        else:
                            continue
                    res_peaks.append(max_index)
                    last_max = max_peak

                accx_peaks_2, _ = find_peaks(df.loc[int((end + last_start)/2):end, feat_name], height=feat_threshold)
                max_peak = 0
                max_index = 0

                for i in accx_peaks_2:
                    if abs(df.at[last_start + i, feat_name]) > max_peak:
                        max_peak = abs(df.at[last_start + i, feat_name])
                        max_index = last_start + i
                if max_index != 0:
                    if res_peaks and max_index - res_peaks[-1] <= int(window_size * 1.3):
                        if max_peak > last_max:
                            res_peaks.pop()
                            res_peaks.append(max_index)
                        else:
                            continue
                    res_peaks.append(max_index)
                    last_max = max_peak

            else:
                accx_peaks, _ = find_peaks(df.iloc[last_start:end][feat_name], height=feat_threshold)
                max_peak = 0
                max_index = 0

                for i in accx_peaks:
                    if abs(df.at[last_start + i, feat_name]) > max_peak:
                        max_peak = abs(df.at[last_start + i, feat_name])
                        max_index = last_start + i
                if max_index != 0:
                    if res_peaks and max_index - res_peaks[-1] <= int(window_size * 1.3):
                        if max_peak > last_max:
                            res_peaks.pop()
                            res_peaks.append(max_index)
                        else:
                            continue
                    res_peaks.append(max_index)
                    last_max = max_peak

    print(res_peaks)


    # 찾은 피크 시각화
    plt.subplot(3, 1, 1)
    for feat_peak in acc_peaks.values():
        for peak in feat_peak:
            plt.axvspan(int(peak - window_size/2), int(peak + window_size/2), alpha =.1, facecolor='g',edgecolor='black')
    plt.plot(df['AccelerationX'], label='X')
    plt.plot(df['AccelerationY'], label='Y')
    plt.plot(df['AccelerationZ'], label='Z')
    plt.title('Acceleration Data')
    plt.xlabel('Timestamp')
    plt.ylabel('Values')
    plt.legend()

    plt.subplot(3, 1, 2)
    for feat_peak in rot_peaks.values():
        for peak in feat_peak:
            plt.axvspan(int(peak - window_size/2), int(peak + window_size/2), alpha =.1, facecolor='g',edgecolor='black')
    plt.plot(df['RRX'], label='p')
    plt.plot(df['RRY'], label='w')
    plt.plot(df['RRZ'], label='r')
    plt.title('Gyro Data')
    plt.xlabel('Timestamp')
    plt.ylabel('Values')
    plt.legend()

    plt.subplot(3, 1, 3)
    for peak in res_peaks:
        plt.axvspan(int(peak - window_size/2), int(peak + window_size/2), alpha =.1, facecolor='r',edgecolor='black')
    plt.plot(peak_range, label='index')

    plt.show()

    return res_peaks


def makeTest(stroke):
    data_list = getData(stroke)

    window_size = 30
    acc_threshold = 1.5
    rr_threshold = 200
    feat_name = 'AccelerationZ'
    feat_threshold = 1
    peak_threshold = 3

    script_dir = os.path.dirname(os.path.abspath(__file__))

    for key, value in data_list.items():
        print(key)
        peaks = findPeak(value, window_size, acc_threshold, rr_threshold, feat_name, feat_threshold, peak_threshold)

        for peak in peaks:
            X_y = pd.DataFrame(columns=['AccelerationX', 'AccelerationY', 'AccelerationZ', 'Pitch', 'Roll', 'Yaw', 'RRX', 'RRY', 'RRZ'])
            for i in range(int(peak - window_size/2), int(peak + window_size/2)):
                d = pd.DataFrame({'AccelerationX': [value['AccelerationX'][i]],
                    'AccelerationY': [value['AccelerationY'][i]], 
                    'AccelerationZ': [value['AccelerationZ'][i]],
                    'Pitch': [value['Pitch'][i]],
                    'Roll' : [value['Roll'][i]],
                    'Yaw' : [value['Yaw'][i]],
                    'RRX': [value['RRX'][i]],
                    'RRY' : [value['RRY'][i]],
                    'RRZ' : [value['RRZ'][i]]})
                X_y = pd.concat([X_y, d], ignore_index=True)
            
            X_y.to_csv(script_dir + '/test/' + key + str(peak) + '.csv', index=False)

# makeTest('stroke data/backhand drive/')
# makeTest('stroke data/backhand hairpin/')
# makeTest('stroke data/backhand highclear/')
# makeTest('stroke data/backhand underclear/')
# makeTest('stroke data/forehand drive/')
# makeTest('stroke data/forehand drop/')
# makeTest('stroke data/forehand hairpin/')
makeTest('stroke data/forehand highclear/')
# makeTest('stroke data/forehand smash/')
# makeTest('stroke data/forehand underclear/')
# makeTest('stroke data/long service/')
# makeTest('stroke data/short service/')