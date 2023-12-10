import pandas as pd
import json
import csv
import math
import os
import ast
import json

import utils as u
from constant import CONST


def cutData(all_data):
    return_data = []

    for data in all_data:

        sorted_column = data[CONST.ACCZ].sort_values(ascending=False)
        first_largest_value = sorted_column.iloc[0]
        second_largest_value = sorted_column.iloc[1]

        if second_largest_value > (first_largest_value * 4 / 5):
            second_index = data[data[CONST.ACCZ] == second_largest_value].index[0]
            first_index = data[data[CONST.ACCZ] == first_largest_value].index[0]
            if second_index > first_index:
                max_index = second_index
            else:
                max_index = data[CONST.ACCZ].idxmax()
        else:
            max_index = data[CONST.ACCZ].idxmax()

        start_index = max_index - CONST.SWING_BEFORE_IMPACT
        end_index = max_index + CONST.SWING_AFTER_IMPACT

        sliced_df = data.iloc[start_index:end_index].reset_index(drop=True)

        return_data.append(sliced_df)

    return return_data

def simpleCutData(all_data):
    return_data = []

    for data in all_data:
        return_data.append(data.iloc[int(CONST.CLASS_WINDOW_SIZE / 2) - CONST.SWING_BEFORE_IMPACT:int(CONST.CLASS_WINDOW_SIZE / 2) + CONST.SWING_AFTER_IMPACT].reset_index(drop=True))
    
    return return_data

def avgData(all_data):
    sum_df = all_data[0].copy()

    for df in all_data[1:]:
        sum_df += df

    average_df = sum_df / len(all_data)
    
    return [average_df]

def addFeature(all_data):
    for data in all_data:

        data[CONST.RVP] = data[CONST.RVP] * 180 / math.pi
        data[CONST.RVR] = data[CONST.RVR] * 180 / math.pi
        data[CONST.RVY] = data[CONST.RVY] * 180 / math.pi

        data['x+y+z'] = data[CONST.ACCX] + data[CONST.ACCY] + data[CONST.ACCZ]
        data['x+y'] = data[CONST.ACCX] + data[CONST.ACCY]
        data['y+z'] = data[CONST.ACCY] + data[CONST.ACCZ]
        data['x+z'] = data[CONST.ACCX] + data[CONST.ACCZ]

        data['x*y*z'] = data[CONST.ACCX] * data[CONST.ACCY] * data[CONST.ACCZ]
        data['x*y'] = data[CONST.ACCX] * data[CONST.ACCY]
        data['y*z'] = data[CONST.ACCY] * data[CONST.ACCZ]
        data['x*z'] = data[CONST.ACCX] * data[CONST.ACCZ]

        data['p+r+w'] = data[CONST.RVP] + data[CONST.RVR] + data[CONST.RVY]
        data['p+r'] = data[CONST.RVP] + data[CONST.RVR]
        data['r+w'] = data[CONST.RVR] + data[CONST.RVY]
        data['p+w'] = data[CONST.RVP] + data[CONST.RVY]

        data['p*r*w'] = data[CONST.RVP] * data[CONST.RVR] * data[CONST.RVY]
        data['p*r'] = data[CONST.RVP] * data[CONST.RVR]
        data['r*w'] = data[CONST.RVR] * data[CONST.RVY]
        data['p*w'] = data[CONST.RVP] * data[CONST.RVY]

    return all_data

def getAllCSV(name):
    return_data = []

    # expert 폴더 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, 'data/temp')

    # 폴더 내의 모든 파일에 대해 작업 수행
    for file_name in os.listdir(folder_path):

        # 파일 확인
        if file_name.endswith('.csv') and file_name.startswith(name):

            # 파일 경로 설정
            file_path = os.path.join(folder_path, file_name)

            df = pd.read_csv(file_path)
            # import matplotlib.pyplot as plt
            # print(file_name)
            # plt.subplot(2, 1, 1)
            # plt.plot(df.index, df[CONST.ACCX], label='X')
            # plt.plot(df.index, df[CONST.ACCY], label='Y')
            # plt.plot(df.index, df[CONST.ACCZ], label='Z')
            # plt.xlabel('Index')
            # plt.ylabel('acceleration')
            # plt.legend()
# 
            # plt.subplot(2, 1, 2)
            # plt.plot(df.index, df[CONST.RVP], label='Pitch')
            # plt.plot(df.index, df[CONST.RVR], label='Roll')
            # plt.plot(df.index, df[CONST.RVY], label='Yaw')
            # plt.xlabel('Index')
            # plt.ylabel('Gyroscope')
            # plt.legend()
            # plt.show()

            return_data.append(addFeature(simpleCutData([df]))[0])
    
    return return_data

def cutMotion(all_data):
    return_data = []

    for data in all_data:

        start_index = 0
        end_index = CONST.SWING_BEFORE_IMPACT + CONST.SWING_AFTER_IMPACT

        data = data.loc[:, ~data.columns.str.contains('Unnamed')]
        sliced_df = data.iloc[start_index:end_index].reset_index(drop=True)

        return_data.append(sliced_df)

    return return_data

def parse_tuple(string):
    try:
        return ast.literal_eval(string)
    except (SyntaxError, ValueError):
        return None  # 파싱 오류 시 None 반환

def avgMotion(all_data):
    # 데이터프레임의 모든 열에 대해 튜플 값을 파싱
    for i in range(len(all_data)):
        all_data[i] = all_data[i].applymap(parse_tuple)

    sum_df = all_data[0].copy()

    for df in all_data[1:]:
        for col in range(len(df.columns)):
            for row in range(len(df)):
                sum_df.iloc[row, col] = tuple(sum(elem) for elem in zip(sum_df.iloc[row, col], df.iloc[row, col]))

    # 합산한 결과를 데이터프레임의 수로 나누어서 평균 계산
    for df in all_data[1:]:
        for col in range(len(df.columns)):
            for row in range(len(df)):
                sum_df.iloc[row, col] = tuple(elem / len(all_data) for elem in sum_df.iloc[row, col])

    return [sum_df]

def getCSVMotionList(list):
    return_data = []

    for name in list:
        # expert 폴더 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(script_dir, name[0])

        same_name = []

        # 폴더 내의 모든 파일에 대해 작업 수행
        for file_name in os.listdir(folder_path):

            # 파일이 CSV 파일인지 확인
            if file_name.endswith('.csv') and file_name.startswith(name[1]):

                # CSV 파일 경로 설정
                file_path = os.path.join(folder_path, file_name)

                # CSV 파일 읽기
                df = pd.read_csv(file_path)

                same_name.append(cutMotion([df])[0])
        return_data.append(avgMotion(same_name)[0])
    
    return return_data

def addMotion(all_data):
    for data in all_data:
        
        for t in range(len(data)):
                    
            data.at[t, 'right_elbow p right_hip'] = (float(data.at[t, 'right_elbow'][0]) - float(data.at[t, 'right_shoulder'][0])) * (float(data.at[t, 'right_hip'][0]) - float(data.at[t, 'right_shoulder'][0])) + (float(data.at[t, 'right_elbow'][1]) - float(data.at[t, 'right_shoulder'][1])) * (float(data.at[t, 'right_hip'][1]) - float(data.at[t, 'right_shoulder'][1]))
            data.at[t, 'hip distance'] = ((float(data.at[t, 'left_hip'][0]) - float(data.at[t, 'right_hip'][0]))**2 + (float(data.at[t, 'left_hip'][1]) - float(data.at[t, 'right_hip'][1]))**2)**(1/2)

            # data.at[t, data.columns[i] + ' p ' + data.columns[j]] = float(data.iloc[t, i][0]) * float(data.iloc[t, j][1]) + float(data.iloc[t, i][0]) * float(data.iloc[t, j][1])
            # data.at[t, data.columns[i] + ' d ' + data.columns[j]] = ((float(data.iloc[t, i][0]) - float(data.iloc[t, j][1]))**2 + (float(data.iloc[t, i][0]) - float(data.iloc[t, j][1]))**2)**(1/2)
        
        del data['nose']
        del data['left_eye']
        del data['right_eye']
        del data['left_ear']
        del data['right_ear']
        del data['left_shoulder']
        del data['right_shoulder']
        del data['left_elbow']
        del data['right_elbow']
        del data['left_wrist']
        del data['right_wrist']
        del data['left_hip']
        del data['right_hip']
        del data['left_knee']
        del data['right_knee']
        del data['left_ankle']
        del data['right_ankle']

    return all_data

def normalize(all_data):
    return_data = []

    for data in all_data:
        result = data.copy()

        for feature_name in data.columns:
            max_value = data[feature_name].max()
            min_value = data[feature_name].min()

            result[feature_name] = (data[feature_name] - min_value) / (max_value - min_value)
        
        return_data.append(result)
    
    return return_data

def windowing(all_data, size):
    return_data = []

    for data in all_data:
        windowed_df = pd.DataFrame(columns=data.columns.tolist()) 

        for i in range(len(data) - size + 1):
            dic = {}

            for column_name in data.columns:
                dic[column_name] = [data[column_name].iloc[i : i + size].tolist()]

            windowed_df = pd.concat([windowed_df, pd.DataFrame(dic)], ignore_index=True)

        return_data.append(windowed_df)

    return return_data

def euclideanDistance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError('벡터 길이가 같아야 합니다.')
    
    # 각 차원에서 차이를 제곱한 후 더한 값
    sum_of_squares = sum((v1 - v2) ** 2 for v1, v2 in zip(vector1, vector2))
    
    # 제곱근을 취한 결과가 두 벡터 간의 거리
    distance = math.sqrt(sum_of_squares)
    
    return distance

def compareOneDistance(one, all_data):

    avg = one[0]
    amount = len(all_data)
    return_dict = {}

    for feature_name in avg.columns:

        for i in range(len(avg)):
            return_dict[(i, feature_name)] = 0

        for data in all_data:

            for row in range(len(data)):
                now = data.loc[row, feature_name]

                c = []
                c.append(euclideanDistance(now, avg.loc[row - 1, feature_name]) if row > 0 else len(now))
                c.append(euclideanDistance(now, avg.loc[row, feature_name]))
                c.append(euclideanDistance(now, avg.loc[row + 1, feature_name]) if row < (len(data) - 1) else len(now))

                return_dict[(row, feature_name)] = return_dict[(row, feature_name)] + (min(c) / amount)


    return return_dict

def calStd(avg, all_data, distance_avg):

    amount = len(all_data)
    return_dict = {}

    for feature_name in avg[0].columns:

        for i in range(len(avg[0])):
            return_dict[(i, feature_name)] = 0

        for data in all_data:

            for row in range(len(data)):
                now = data.loc[row, feature_name]

                c = []
                c.append(euclideanDistance(now, avg[0].loc[row - 1, feature_name]) if row > 0 else len(now))
                c.append(euclideanDistance(now, avg[0].loc[row, feature_name]))
                c.append(euclideanDistance(now, avg[0].loc[row + 1, feature_name]) if row < (len(data) - 1) else len(now))

                return_dict[(row, feature_name)] = return_dict[(row, feature_name)] + ((min(c) - distance_avg[(row, feature_name)])**2)

    
    return {key: (value / amount)**(1 / 2) for key, value in return_dict.items()}


ex_cut_data = getAllCSV('s ss')

ex_avg = avgData(ex_cut_data)

ex_normalize_data = normalize(ex_cut_data) # ex_add_data
ex_windowed_data = windowing(ex_normalize_data, CONST.SWING_WINDOW_SIZE)

ex_avg_normalize_data = normalize(ex_avg)
ex_avg_windowed_data = windowing(ex_avg_normalize_data, CONST.SWING_WINDOW_SIZE)

# 거리 재기 return {(start_time, feat_name): 거리평균}
# 다른 스윙의 1 만큼 앞뒤로 더 검사해 가장 가까운 거리를 기준으로 return을 계산한다. 시작 timestamp에 결과가 적용된다.
ex_distance = compareOneDistance(ex_avg_windowed_data, ex_windowed_data)

# 거리 평균의 표준 편차
ex_std = calStd(ex_avg_windowed_data, ex_windowed_data, ex_distance)

# 전문가 평균, 표준 편차 csv로 저장 시작
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_name = '/ex_avg.csv'
ex_avg_windowed_data[0].to_csv(script_dir + csv_file_name, index=False)

# (6, 'hip distance'): 0.47564265211011747 -> 6, 'hip distance', 0.47564265211011747
rows = [(key[0], key[1], value) for key, value in ex_std.items()]
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_name = '/ex_std.csv'
with open(script_dir + csv_file_name, 'w', newline='') as csvfile:
    fieldnames = [CONST.TIMESTAMP, CONST.FEATURE, CONST.VALUE]
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)
    writer.writerows(rows)

rows = [(key[0], key[1], value) for key, value in ex_distance.items()]
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_name = '/ex_dist.csv'
with open(script_dir + csv_file_name, 'w', newline='') as csvfile:
    fieldnames = [CONST.TIMESTAMP, CONST.FEATURE, CONST.VALUE]
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)
    writer.writerows(rows)