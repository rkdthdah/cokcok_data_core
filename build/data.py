import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import math
import os
import ast
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

def getData(forder):
    return_data = []

    # expert 폴더 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, forder)

    # 폴더 내의 모든 파일에 대해 작업 수행
    for file_name in os.listdir(folder_path):

        # 파일이 JSON 파일인지 확인
        if file_name.endswith('.json'):

            # JSON 파일 경로 설정
            file_path = os.path.join(folder_path, file_name)

            # JSON 파일 읽기
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)

            # 빈 리스트 생성
            new_data = []

            # JSON 데이터를 원하는 형태로 변환
            for entry in data:
                new_entry = {
                    "timestamp": entry["timestamp"],
                    "x": float(entry["userAcceleration"]["x"]),
                    "y": float(entry["userAcceleration"]["y"]),
                    "z": float(entry["userAcceleration"]["z"]),
                    "p": np.sin(float(entry["attitude"]["pitch"])),
                    "r": np.sin(float(entry["attitude"]["roll"])),
                    "w": np.sin(float(entry["attitude"]["yaw"]))
                }
                new_data.append(new_entry)

            # 변환된 데이터를 Pandas DataFrame으로 변환
            df = pd.DataFrame(new_data)

            return_data.append(df)
    
    return return_data

def getOne(forder, name):
    return_data = []
    
    # expert 폴더 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, forder)

        # 파일이 JSON 파일인지 확인
    file_name = name + ".json"

    # JSON 파일 경로 설정
    file_path = os.path.join(folder_path, file_name)

    # JSON 파일 읽기
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    # 빈 리스트 생성
    new_data = []

    # JSON 데이터를 원하는 형태로 변환
    for entry in data:
        new_entry = {
            "timestamp": entry["timestamp"],
            "x": float(entry["userAcceleration"]["x"]),
            "y": float(entry["userAcceleration"]["y"]),
            "z": float(entry["userAcceleration"]["z"]),
            "p": np.sin(float(entry["attitude"]["pitch"])),
            "r": np.sin(float(entry["attitude"]["roll"])),
            "w": np.sin(float(entry["attitude"]["yaw"]))
        }
        new_data.append(new_entry)

    # 변환된 데이터를 Pandas DataFrame으로 변환
    df = pd.DataFrame(new_data)

    return_data.append(df)
    
    return return_data

def makeGraph(forder):
    # expert 폴더 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, forder)

    # 폴더 내의 모든 파일에 대해 작업 수행
    for file_name in os.listdir(folder_path):

        # 파일이 JSON 파일인지 확인
        if file_name.endswith('.json'):

            # JSON 파일 경로 설정
            file_path = os.path.join(folder_path, file_name)

            # JSON 파일 읽기
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)

            # 빈 리스트 생성
            new_data = []

            # JSON 데이터를 원하는 형태로 변환
            for entry in data:
                new_entry = {
                    "timestamp": entry["timestamp"],
                    "x_acceleration": float(entry["userAcceleration"]["x"]),
                    "y_acceleration": float(entry["userAcceleration"]["y"]),
                    "z_acceleration": float(entry["userAcceleration"]["z"]),
                    "roll": float(entry["attitude"]["roll"]),
                    "yaw": float(entry["attitude"]["yaw"]),
                    "pitch": float(entry["attitude"]["pitch"])
                }
                new_data.append(new_entry)

            # 변환된 데이터를 Pandas DataFrame으로 변환
            df = pd.DataFrame(new_data)
            
            # 그래프 생성 및 저장
            plt.figure(figsize=(10, 6))

            plt.subplot(2, 1, 1)
            plt.plot(df['timestamp'], np.sin(df['pitch']), label='Sin(Pitch)')
            plt.plot(df['timestamp'], np.sin(df['roll']), label='Sin(Roll)')
            plt.plot(df['timestamp'], np.sin(df['yaw']), label='Sin(Yaw)')
            plt.title('Sin Transformed Attitude Data')
            plt.xlabel('Timestamp')
            plt.ylabel('Values')
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(df['timestamp'], df['x_acceleration'], label='X')
            plt.plot(df['timestamp'], df['y_acceleration'], label='Y')
            plt.plot(df['timestamp'], df['z_acceleration'], label='Z')
            plt.title('User Acceleration Data')
            plt.xlabel('Timestamp')
            plt.ylabel('Values')
            plt.legend()

            # 그래프 파일로 저장 (파일 이름은 JSON 파일 이름과 동일하게)
            graph_file_name = os.path.splitext(file_name)[0] + '.png'
            graph_file_path = os.path.join(folder_path, graph_file_name)
            plt.tight_layout()
            plt.savefig(graph_file_path)
            plt.close()
    
    return

def cutData(all_data):
    return_data = []

    for data in all_data:

        sorted_column = data['z'].sort_values(ascending=False)
        first_largest_value = sorted_column.iloc[0]
        second_largest_value = sorted_column.iloc[1]

        if second_largest_value > (first_largest_value * 9 / 10):
            second_index = data[data['z'] == second_largest_value].index[0]
            first_index = data[data['z'] == first_largest_value].index[0]
            if second_index > first_index:
                max_index = second_index
            else:
                max_index = data['z'].idxmax()
        else:
            max_index = data['z'].idxmax()

        start_index = max_index - 10
        end_index = max_index + 4

        sliced_df = data.iloc[start_index:end_index].reset_index(drop=True)

        return_data.append(sliced_df)

    return return_data

def cutMotion(all_data):
    return_data = []

    for data in all_data:

        start_index = 0
        end_index = 14

        data = data.loc[:, ~data.columns.str.contains('Unnamed')]
        sliced_df = data.iloc[start_index:end_index].reset_index(drop=True)

        return_data.append(sliced_df)

    return return_data

def addFeature(all_data):
    for data in all_data:

        for i in range(1, len(data)):
            if data.at[i, 'p'] < (data.at[i - 1, 'p'] - 3): data.at[i, 'p'] += 2 * math.pi
            if data.at[i, 'r'] < (data.at[i - 1, 'r'] - 3): data.at[i, 'r'] += 2 * math.pi
            if data.at[i, 'w'] < (data.at[i - 1, 'w'] - 3): data.at[i, 'w'] += 2 * math.pi
            
            if data.at[i, 'p'] > (data.at[i - 1, 'p'] + 3): data.at[i, 'p'] -= 2 * math.pi
            if data.at[i, 'r'] > (data.at[i - 1, 'r'] + 3): data.at[i, 'r'] -= 2 * math.pi
            if data.at[i, 'w'] > (data.at[i - 1, 'w'] + 3): data.at[i, 'w'] -= 2 * math.pi


        data['dw'] = data['w'] - data['w'].shift(1)
        data['dw'] = data['dw'].fillna(0)

        del data['w']

        data['x+y+z'] = data['x'] + data['y'] + data['z']
        data['x+y'] = data['x'] + data['y']
        data['y+z'] = data['y'] + data['z']
        data['x+z'] = data['x'] + data['z']

        data['x*y*z'] = data['x'] * data['y'] * data['z']
        data['x*y'] = data['x'] * data['y']
        data['y*z'] = data['y'] * data['z']
        data['x*z'] = data['x'] * data['z']
        
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
        df = pd.DataFrame(data).drop('timestamp', axis=1)
        windowed_df = pd.DataFrame(columns=['x', 'y', 'z', 'p', 'r', 'dw'])

        for i in range(len(data) - size + 1):
            dic = {}

            for column_name in df.columns:
                dic[column_name] = [df[column_name].iloc[i : i + size].tolist()]

            windowed_df = pd.concat([windowed_df, pd.DataFrame(dic)], ignore_index=True)

        return_data.append(windowed_df)

    return return_data

def euclideanDistance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("벡터 길이가 같아야 합니다.")
    
    # 각 차원에서 차이를 제곱한 후 더한 값
    sum_of_squares = sum((v1 - v2) ** 2 for v1, v2 in zip(vector1, vector2))
    
    # 제곱근을 취한 결과가 두 벡터 간의 거리
    distance = math.sqrt(sum_of_squares)
    
    return distance

def compareEachDistance(all_data):

    amount = len(all_data) * (len(all_data) - 1)
    return_dict = {}

    for feature_name in all_data[0].columns:

        for i in range(len(all_data[0])):
            return_dict[(i, feature_name)] = 0

        for data_index, data in enumerate(all_data):

            for row in range(len(data)):
                now = data.loc[row, feature_name]

                for compare_index in range(len(all_data)):
                    if compare_index == data_index: continue

                    c = []
                    c.append(euclideanDistance(now, all_data[compare_index].loc[row - 1, feature_name]) if row > 0 else len(now))
                    c.append(euclideanDistance(now, all_data[compare_index].loc[row, feature_name]))
                    c.append(euclideanDistance(now, all_data[compare_index].loc[row + 1, feature_name]) if row < (len(data) - 1) else len(now))

                    return_dict[(row, feature_name)] = return_dict[(row, feature_name)] + (min(c) / amount)

    return return_dict

def avgData(all_data):
    sum_df = all_data[0].copy()

    for df in all_data[1:]:
        sum_df += df

    average_df = sum_df / len(all_data)
    
    return [average_df]

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

def getPersonList(list):
    return_data = []

    for name in list:
        # expert 폴더 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(script_dir, name[0])

        same_name = []

        # 폴더 내의 모든 파일에 대해 작업 수행
        for file_name in os.listdir(folder_path):

            # 파일이 JSON 파일인지 확인
            if file_name.endswith('.json') and file_name.startswith(name[1]):

                # JSON 파일 경로 설정
                file_path = os.path.join(folder_path, file_name)

                # JSON 파일 읽기
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)

                # 빈 리스트 생성
                new_data = []

                # JSON 데이터를 원하는 형태로 변환
                for entry in data:
                    new_entry = {
                        "timestamp": entry["timestamp"],
                        "x": float(entry["userAcceleration"]["x"]),
                        "y": float(entry["userAcceleration"]["y"]),
                        "z": float(entry["userAcceleration"]["z"]),
                        "p": float(entry["attitude"]["pitch"]),
                        "r": float(entry["attitude"]["roll"]),
                        "w": float(entry["attitude"]["yaw"])
                    }
                    new_data.append(new_entry)

                # 변환된 데이터를 Pandas DataFrame으로 변환
                df = pd.DataFrame(new_data)

                same_name.append(cutData([df])[0])
        
        return_data.append(avgData(same_name)[0])
    
    return return_data

def getPersonMotionList(list):
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
                #c.append(euclideanDistance(now, avg[0].loc[row - 1, feature_name]) if row > 0 else len(now))
                c.append(euclideanDistance(now, avg[0].loc[row, feature_name]))
                #c.append(euclideanDistance(now, avg[0].loc[row + 1, feature_name]) if row < (len(data) - 1) else len(now))

                return_dict[(row, feature_name)] = return_dict[(row, feature_name)] + ((min(c) - distance_avg[(row, feature_name)])**2)

    
    return {key: (value / amount)**(1 / 2) for key, value in return_dict.items()}

def addMotion(all_data):
    for data in all_data:
        
        for t in range(len(data)):
                    
            data.at[t, "right_elbow p right_hip"] = (float(data.at[t, 'right_elbow'][0]) - float(data.at[t, 'right_shoulder'][0])) * (float(data.at[t, 'right_hip'][0]) - float(data.at[t, 'right_shoulder'][0])) + (float(data.at[t, 'right_elbow'][1]) - float(data.at[t, 'right_shoulder'][1])) * (float(data.at[t, 'right_hip'][1]) - float(data.at[t, 'right_shoulder'][1]))
            data.at[t, "hip distance"] = ((float(data.at[t, 'left_hip'][0]) - float(data.at[t, 'right_hip'][0]))**2 + (float(data.at[t, 'left_hip'][1]) - float(data.at[t, 'right_hip'][1]))**2)**(1/2)

            # data.at[t, data.columns[i] + " p " + data.columns[j]] = float(data.iloc[t, i][0]) * float(data.iloc[t, j][1]) + float(data.iloc[t, i][0]) * float(data.iloc[t, j][1])
            # data.at[t, data.columns[i] + " d " + data.columns[j]] = ((float(data.iloc[t, i][0]) - float(data.iloc[t, j][1]))**2 + (float(data.iloc[t, i][0]) - float(data.iloc[t, j][1]))**2)**(1/2)
        
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

window_size = 8

ex_list = [["expert", "A남10"], 
            ["expert", "A남11"], 
            ["expert", "A남13"]]
ex_motion_list = [["expertmotion", "A남10"], 
            ["expertmotion", "A남11"], 
            ["expertmotion", "A남13"]]
# data 읽기 및 dataframe으로 변환
ex_cut_data = getPersonList(ex_list)
ex_motion_cut_data = getPersonMotionList(ex_motion_list)

# 이 부분에 차원 추가가 들어간다. def 차원추가(cutted_data) -> def normalize(it)
ex_add_data = addFeature(ex_cut_data)
ex_add_motion = addMotion(ex_motion_cut_data)

ex_add_all = []
for i in range(len(ex_add_data)):
    ex_add_all.append(pd.concat([ex_add_data[i], ex_add_motion[i]], axis=1))

# 전문가 평균
ex_avg = avgData(ex_add_all)

ex_normalize_data = normalize(ex_add_all)
ex_windowed_data = windowing(ex_normalize_data, window_size)

ex_avg_normalize_data = normalize(ex_avg)
ex_avg_windowed_data = windowing(ex_avg_normalize_data, window_size)
# print("본데이터")
# print(ex_windowed_data[0].loc[6, 'x'])
# print("평균")
# print(ex_avg_windowed_data[0].loc[6, 'x'])
# ex_std = calStd(ex_avg_windowed_data, ex_windowed_data)
# print("분산")
# print(ex_std)

# 거리 재기 return {(start_time, feat_name): 거리평균}
# 다른 스윙의 1 만큼 앞뒤로 더 검사해 가장 가까운 거리를 기준으로 return을 계산한다. 시작 timestamp에 결과가 적용된다.
ex_distance = compareOneDistance(ex_avg_windowed_data, ex_windowed_data)
# 거리 평균의 표준 편차
ex_std = calStd(ex_avg_windowed_data, ex_windowed_data, ex_distance)

print(sorted(ex_std.items(), key=lambda x: x[1]))

"""
# data 읽기 및 dataframe으로 변환
wg_data = getData("wg")
ulsan_data = getData("intermediate")
low_data = wg_data + ulsan_data
#low_data = getData("wg")

# data 전처리 
low_cut_data = cutData(low_data)

# 이 부분에 차원 추가가 들어간다. def 차원추가(cutted_data) -> def normalize(it)
ex_add_data = addFeature(low_cut_data)

low_normalize_data = normalize(ex_add_data)
low_windowed_data = windowing(low_normalize_data, window_size)

low_distance = compareOneDistance(ex_avg_windowed_data, low_windowed_data)


# 일반인 data / 전문가 data
result_dict = {key: low_distance[key] / ex_distance[key] for key in ex_distance if key in low_distance}

max_value = max(result_dict.values())

# 실력 평가에서 각 feature에 대한 가중치가 될 score_dict
score_dict = {key: 0 for key in ex_distance.keys()}
for key, value in result_dict.items():
    if value < 1:
        score_dict[key] = 0
    else:
        score_dict[key] += value / max_value

# 비전문 데이터에서 무작위 1/3개 추출
# for i in range(len(low_data)//3):
#     random_sample = random.sample(low_windowed_data, len(low_data)//3)

#     # 거리 재기 return {(start_time, feat_name): 거리평균}
#     # 다른 스윙의 1 만큼 앞뒤로 더 검사해 가장 가까운 거리를 기준으로 return을 계산한다. 시작 timestamp에 결과가 적용된다.
#     low_weight = compareOneDistance(ex_avg_windowed_data, random_sample)


#     result_dict = {key: low_weight[key] / ex_weight[key] for key in ex_weight if key in low_weight}

#     for key, value in result_dict.items():
#         score_dict[key] += value / (len(low_data)//3)


sorted_score = sorted(score_dict.items(), key=lambda x: x[1])
"""



nameList1 = [["expert", "A남10"], 
             ["expert", "A남11"], 
             ["expert", "A남13"], 
            ["intermediate", "C남10"], 
            ["intermediate", "C남2년"],
            ["intermediate", "C남1년"],
            ["intermediate", "D남1."],
            ["intermediate", "D여3년"],
            ["intermediate", "D남1년"],
            ["intermediate", "D여6개"]]

nameList2 = [["wg", "강송모"],
            ["wg", "권윤호"],
            #["wg", "김도규"],
            ##["wg", "김민주"],
            ["wg", "김송은"],
            #["wg", "김유영"],
            ["wg", "박소현"],
            ["wg", "박지환"],
            ##["wg", "서예은"],
            ##["wg", "이가은"],
            ["wg", "이신우"],
            ["wg", "이윤신"],
            ##["wg", "정보석"],
            ["wg", "정석훈"],
            ["wg", "최지웅"],
            ["wg", "신도영"],
            ["wg", "이태경"]
            ]

nameList2_mo = [["wgmotion", "강송모"],
            ["wgmotion", "권윤호"],
            #["wgmotion", "김도규"],
            ##["wgmotion", "김민주"],
            ["wgmotion", "김송은"],
            #["wgmotion", "김유영"],
            ["wgmotion", "박소현"],
            ["wgmotion", "박지환"],
            ##["wgmotion", "서예은"],
            ##["wgmotion", "이가은"],
            ["wgmotion", "이신우"],
            ["wgmotion", "이윤신"],
            ##["wgmotion", "정보석"],
            ["wgmotion", "정석훈"],
            ["wgmotion", "최지웅"],
            ["wgmotion", "신도영"],
            ["wgmotion", "이태경"]
            ]

notice = 0

#nameList = nameList1 + nameList2
nameList = nameList2

#personList = getPersonList(nameList1)
personList = getPersonList(nameList2)
motionList = getPersonMotionList(nameList2_mo)

#personList = personList + getPersonList(nameList2)
added_feature_test = addFeature(personList)
added_motion_test = addMotion(motionList)

test_add_all = []
for i in range(len(added_motion_test)):
    test_add_all.append(pd.concat([added_feature_test[i], added_motion_test[i]], axis=1))

print(len(added_feature_test))
print(len(added_motion_test))
print(len(test_add_all))

windowed_data = windowing(normalize(test_add_all), window_size)

distance = []
for data in windowed_data:
    distance.append(compareOneDistance(ex_avg_windowed_data, [data]))

result = [0] * len(distance)

# for i, dis in enumerate(distance):
#     for key in score_dict.keys():
#         result[i] += score_dict[key] * dis[key]

# for i, dis in enumerate(distance):
#     for key in distance[0].keys():
#         result[i] += dis[key]

# for i, dis in enumerate(distance):
#     for key in distance[0].keys():
#         dissum = dis[key] - ex_distance[key]
#         if dissum < 0:
#             result[i] += 0
#         else:
#             result[i] += dissum

max11 = []

for i, dis in enumerate(distance):
    for key in distance[9].keys():
        dissum = (dis[key] - ex_distance[key]) / ex_std[key]
        dissumno = (dis[key] - ex_distance[key]) / ex_std[key]
        if dissum > 0:
            result[i] += dissum
        if i == notice:
            if len(max11) < 16:
                max11.append((key, dissumno))
            else:
                max11.remove(min(max11, key=lambda x: x[1]))
                max11.append((key, dissumno))


rank_dict = {}

for i in range(len(result)):
    rank_dict[nameList[i][1]] = result[i]

sorted_rank = sorted(rank_dict.items(), key=lambda x: x[1])

for value in sorted_rank:
    print(value)

actual_intermediate = ["이윤신", "정보석", "권윤호", "신도영", "강송모", "정석훈", "이태경"]
actual_novice = ["박지환", "최지웅", "이신우", "김민주", "서예은", "박소현", "김송은", "이가은"]

actual_all = actual_intermediate + actual_novice

actual_all_motion = ["이윤신", "권윤호", "신도영", "강송모", "정석훈", "이태경", "박지환", "최지웅", "이신우", "박소현", "김송은"]
#actual_all_motion = ["이윤신", "권윤호", "강송모", "정석훈", "박지환", "최지웅", "이신우", "박소현", "김송은"]

position_difference = {}
    
# 이름을 key로 사용하여 튜플 리스트의 위치를 저장
tuple_positions = {item[0]: i for i, item in enumerate(sorted_rank)}

for name in actual_all_motion:
    if name in tuple_positions:
        position_difference[name] = tuple_positions[name] - actual_all_motion.index(name)

difavg = 0
for name, difference in position_difference.items():
    difavg += abs(difference) / len(position_difference)

difavg2 = 0
for name, difference in position_difference.items():
    difavg2 += abs(difference)**2 / len(position_difference)

print(difavg)
print(difavg2)
sorted_max11 = sorted(max11, key=lambda x: abs(x[1]), reverse=True)
for item in sorted_max11:
    print(item)

# correlation, p_value = spearmanr(actual_intermediate + actual_novice, [item[0] for item in sorted_rank])

# print(actual_intermediate)
# print(actual_novice)

# print(f"Spearman 상관 계수: {correlation}")
# print(f"P-값: {p_value}")


graph_data = ex_avg_normalize_data
#tlist = [0, 1, 2, 3, 4, 9] # for wg
tlist = [0, 1, 3, 5] # for expert
t = tlist[0]
plt.subplot(3, 1, 1)
plt.plot(graph_data[t].index, graph_data[t]['x'], label='X')
plt.plot(graph_data[t].index, graph_data[t]['y'], label='Y')
plt.plot(graph_data[t].index, graph_data[t]['z'], label='Z')
plt.xlabel('Index')
plt.ylabel('acceleration')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(graph_data[t].index, graph_data[t]['p'], label='Pitch')
plt.plot(graph_data[t].index, graph_data[t]['r'], label='Roll')
plt.plot(graph_data[t].index, graph_data[t]['y'], label='Yaw')
plt.xlabel('Index')
plt.ylabel('Gyroscope')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(graph_data[t].index, graph_data[t]['right_elbow p right_hip'], label='right_elbow · right_hip')
plt.plot(graph_data[t].index, graph_data[t]['x+z'], label='x+z')
plt.xlabel('Index')
plt.ylabel('motion')
plt.legend()

'''
t = tlist[1]
plt.subplot(12, 1, 2)
plt.plot(graph_data[t].index, graph_data[t]['x'], label='X')
plt.plot(graph_data[t].index, graph_data[t]['y'], label='Y')
plt.plot(graph_data[t].index, graph_data[t]['z'], label='Z')
plt.xlabel('Index')
plt.ylabel('acceleration')
plt.legend()

plt.subplot(12, 1, 8)
plt.plot(graph_data[t].index, graph_data[t]['p'], label='Sin(Pitch)')
plt.plot(graph_data[t].index, graph_data[t]['r'], label='Sin(Roll)')
plt.plot(graph_data[t].index, graph_data[t]['y'], label='Sin(Yaw)')
plt.xlabel('Index')
plt.ylabel('Gyroscope')
plt.legend()

t = tlist[2]
plt.subplot(12, 1, 3)
plt.plot(graph_data[t].index, graph_data[t]['x'], label='X')
plt.plot(graph_data[t].index, graph_data[t]['y'], label='Y')
plt.plot(graph_data[t].index, graph_data[t]['z'], label='Z')
plt.xlabel('Index')
plt.ylabel('acceleration')
plt.legend()

plt.subplot(12, 1, 9)
plt.plot(graph_data[t].index, graph_data[t]['p'], label='Sin(Pitch)')
plt.plot(graph_data[t].index, graph_data[t]['r'], label='Sin(Roll)')
plt.plot(graph_data[t].index, graph_data[t]['y'], label='Sin(Yaw)')
plt.xlabel('Index')
plt.ylabel('Gyroscope')
plt.legend()

t = tlist[3]
plt.subplot(12, 1, 4)
plt.plot(graph_data[t].index, graph_data[t]['x'], label='X')
plt.plot(graph_data[t].index, graph_data[t]['y'], label='Y')
plt.plot(graph_data[t].index, graph_data[t]['z'], label='Z')
plt.xlabel('Index')
plt.ylabel('acceleration')
plt.legend()

plt.subplot(12, 1, 10)
plt.plot(graph_data[t].index, graph_data[t]['p'], label='Sin(Pitch)')
plt.plot(graph_data[t].index, graph_data[t]['r'], label='Sin(Roll)')
plt.plot(graph_data[t].index, graph_data[t]['y'], label='Sin(Yaw)')
plt.xlabel('Index')
plt.ylabel('Gyroscope')
plt.legend()

t = tlist[4]
plt.subplot(12, 1, 5)
plt.plot(graph_data[t].index, graph_data[t]['x'], label='X')
plt.plot(graph_data[t].index, graph_data[t]['y'], label='Y')
plt.plot(graph_data[t].index, graph_data[t]['z'], label='Z')
plt.xlabel('Index')
plt.ylabel('acceleration')
plt.legend()

plt.subplot(12, 1, 11)
plt.plot(graph_data[t].index, graph_data[t]['p'], label='Sin(Pitch)')
plt.plot(graph_data[t].index, graph_data[t]['r'], label='Sin(Roll)')
plt.plot(graph_data[t].index, graph_data[t]['y'], label='Sin(Yaw)')
plt.xlabel('Index')
plt.ylabel('Gyroscope')
plt.legend()

t = tlist[5]
plt.subplot(12, 1, 6)
plt.plot(graph_data[t].index, graph_data[t]['x'], label='X')
plt.plot(graph_data[t].index, graph_data[t]['y'], label='Y')
plt.plot(graph_data[t].index, graph_data[t]['z'], label='Z')
plt.xlabel('Index')
plt.ylabel('acceleration')
plt.legend()

plt.subplot(12, 1, 12)
plt.plot(graph_data[t].index, graph_data[t]['p'], label='Sin(Pitch)')
plt.plot(graph_data[t].index, graph_data[t]['r'], label='Sin(Roll)')
plt.plot(graph_data[t].index, graph_data[t]['y'], label='Sin(Yaw)')
plt.xlabel('Index')
plt.ylabel('Gyroscope')
plt.legend()
'''
plt.show()


    #         timestamps = []
    #         sin_pitch_values = []  # Sin 변환된 pitch 데이터를 저장할 리스트
    #         sin_roll_values = []  # Sin 변환된 roll 데이터를 저장할 리스트
    #         sin_yaw_values = []  # Sin 변환된 yaw 데이터를 저장할 리스트
    #         user_acceleration_x = []
    #         user_acceleration_y = []
    #         user_acceleration_z = []

    #         # 데이터 추출
    #         for entry in data:
    #             timestamps.append(entry["timestamp"])
    #             attitude = entry["attitude"]
    #             pitch = float(attitude["pitch"])
    #             roll = float(attitude["roll"])
    #             yaw = float(attitude["yaw"])

    #             # 각도 데이터를 사인 함수로 변환하여 저장
    #             sin_pitch_values.append(np.sin(pitch))
    #             sin_roll_values.append(np.sin(roll))
    #             sin_yaw_values.append(np.sin(yaw))

    #             user_acceleration = entry["userAcceleration"]
    #             user_acceleration_x.append(float(user_acceleration["x"]))
    #             user_acceleration_y.append(float(user_acceleration["y"]))
    #             user_acceleration_z.append(float(user_acceleration["z"]))

    #         # 그래프 생성 및 저장
    #         plt.figure(figsize=(10, 6))

    #         plt.subplot(2, 1, 1)
    #         plt.plot(timestamps, sin_pitch_values, label='Sin(Pitch)')
    #         plt.plot(timestamps, sin_roll_values, label='Sin(Roll)')
    #         plt.plot(timestamps, sin_yaw_values, label='Sin(Yaw)')
    #         plt.title('Sin Transformed Attitude Data')
    #         plt.xlabel('Timestamp')
    #         plt.ylabel('Values')
    #         plt.legend()

    #         plt.subplot(2, 1, 2)
    #         plt.plot(timestamps, user_acceleration_x, label='X')
    #         plt.plot(timestamps, user_acceleration_y, label='Y')
    #         plt.plot(timestamps, user_acceleration_z, label='Z')
    #         plt.title('User Acceleration Data')
    #         plt.xlabel('Timestamp')
    #         plt.ylabel('Values')
    #         plt.legend()

    #         # 그래프 파일로 저장 (파일 이름은 JSON 파일 이름과 동일하게)
    #         # graph_file_name = os.path.splitext(file_name)[0] + '.png'
    #         # graph_file_path = os.path.join(folder_path, graph_file_name)
    #         # plt.tight_layout()
    #         # plt.savefig(graph_file_path)
    #         # plt.close()

    # # 그래프 창 띄우기
    # plt.tight_layout()
    # plt.show()
    # return
