import pandas as pd
import math
import ast
import csv

from . import exception as ex
from .constant import CONST


class SwingAnalysis:
    """
    스위 데이터로부터 스윙 분석 class

    constant, exception, util 모듈,
    스윙 종류에 따라 {stroke_type}/(ex_avg & ex_dist & ex_std) 파일에 의존성을 갖는다

    field:
    data (DataFrame): 분석 대상 data. 클래스 생성자로 배정된다
    stroke (String): 분석 대상 스윙의 스트로크 종류
    score (Float): 분석 결과 산출된 스윙 점수
    max (Tuple[]): 분석 결과 산출된 오차 리스트. (key = (start index, feature name)), z score)[] 형태. z xcore 기준 내림차순 정렬되어 있다
    res (float[]): 각 timestamp당 오차 점수 합
    res_prepare (float): 준비동작 구간에서의 오차 평균
    res_impact (float): 임펙트 구간에서의 오차 평균
    res_follow (float): 팔로스로우 구간에서의 오차 평균
    res_max (float): 스윙 전체에서 가속도 최고값
    res_feedback (String[]): 각 스윙 구간 별 피드백 key 값 리스트. 피드백 딕셔너리는 constant에 정의

    method:
    __init__ (df): 전역변수 data 배정
    cutData(all_data): 데이터프레임 data로부터 피크를 찾아 적절히 스윙 부분 자르기
    normalize(all_data): 데이터프레임 data 정규화
    windowing(all_data, size=CONST.SWING_WINDOW_SIZE): 데이터프레임 data에 windowing 적용
    euclideanDistance(vector1, vector2): 두 벡터간 유클리디안 거리 산출
    calDistance(avg, all_data): avg와 all_data간 거리 산출. 함수 euclideanDistance를 사용
    analysis(stroke, sol): stroke인 스윙 분석. classification에서 호출될 경우를 제외하고 매개변수 sol=True 이다. 전역변수 stroke, score, max를 배정한다. 앞서 나열한 모든 함수를 사용한다. 스윙 종류에 따라 {stroke_type}/(ex_avg & ex_dist & ex_std) 파일에 의존성을 갖는다
    interpret(self): 스윙 분석 결과로부터 보고서 산출. 전역변수 res를 배정한다. 함수 analysis에 후행한다
    """

    def __init__(self, df):
            self.data = df
            if self.data.empty:
                raise ex.EmptyDataError
            if not type(df) == pd.DataFrame:
                raise ex.InvalidDataError
    
    def addFeature(self, all_data):
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
            
            del data[CONST.PITCH]
            del data[CONST.ROLL]
            del data[CONST.YAW]

        return all_data

    def cutData(self, all_data):
        return_data = []

        for data in all_data:
            sorted_column = data[CONST.ACCZ].sort_values(ascending=False)
            first_largest_value = sorted_column.iloc[0]
            second_largest_value = sorted_column.iloc[1]

            if second_largest_value > (first_largest_value * 9 / 10):
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

    def simpleCutData(self, all_data):
        return_data = []

        for data in all_data:
            return_data.append(data.iloc[int(CONST.CLASS_WINDOW_SIZE / 2) - CONST.SWING_BEFORE_IMPACT:int(CONST.CLASS_WINDOW_SIZE / 2) + CONST.SWING_AFTER_IMPACT].reset_index(drop=True))
        
        return return_data

    def normalize(self, all_data):
        return_data = []

        for data in all_data:
            result = data.copy()

            for feature_name in data.columns:
                max_value = data[feature_name].max()
                min_value = data[feature_name].min()

                result[feature_name] = (data[feature_name] - min_value) / (max_value - min_value)
            
            return_data.append(result)
        
        return return_data

    def windowing(self, all_data, size=CONST.SWING_WINDOW_SIZE):
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
    
    def euclideanDistance(self, vector1, vector2):
        if len(vector1) != len(vector2):
            raise ValueError("벡터 길이가 같아야 합니다.")
        
        # 각 차원에서 차이를 제곱한 후 더한 값
        sum_of_squares = sum((v1 - v2) ** 2 for v1, v2 in zip(vector1, vector2))
        
        # 제곱근을 취한 결과가 두 벡터 간의 거리
        distance = math.sqrt(sum_of_squares)
        
        return distance

    def calDistance(self, avg, all_data):
        amount = len(all_data)
        return_dict = {}

        for feature_name in avg.columns:

            for i in range(len(avg)):
                return_dict[(i, feature_name)] = 0

            for data in all_data:
                for row in range(len(data)):
                    now = data.loc[row, feature_name]

                    c = []
                    c.append(self.euclideanDistance(now, avg.loc[row - 1, feature_name]) if row > 0 else len(now))
                    c.append(self.euclideanDistance(now, avg.loc[row, feature_name]))
                    c.append(self.euclideanDistance(now, avg.loc[row + 1, feature_name]) if row < (len(data) - 1) else len(now))

                    return_dict[(row, feature_name)] = return_dict[(row, feature_name)] + (min(c) / amount)

        return return_dict

    def analysis(self, stroke, sol=True):
        try:
            self.stroke = stroke
            if not self.stroke:
                raise ex.EmptyValueError(self.stroke)
            if not type(self.stroke) == str:
                raise ex.InvalidValueError(self.stroke)

            if sol: self.data = self.cutData([self.data])[0]
            else: self.data = self.simpleCutData([self.data])[0]

            if len(self.data) != CONST.SWING_BEFORE_IMPACT + CONST.SWING_AFTER_IMPACT:
                raise ex.CutDataError(self.data)

            added_data = self.addFeature([self.data])

            self.raw_data = added_data[0].copy()

            user_data = self.windowing(self.normalize(added_data))

            ex_avg_path = CONST.CURRENT_PATH + '/' + stroke + '/ex_avg.csv'
            ex_avg_data = pd.read_csv(ex_avg_path)
            for col in ex_avg_data.columns.to_list():
                ex_avg_data[col] = ex_avg_data[col].apply(ast.literal_eval)
            
            distance = self.calDistance(ex_avg_data, user_data)

            ex_std_data = {}
            ex_std_path = CONST.CURRENT_PATH + '/' + stroke + '/ex_std.csv'
            with open(ex_std_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)

                for row in reader:
                    key = (int(row[CONST.TIMESTAMP]), row[CONST.FEATURE])
                    value = float(row[CONST.VALUE])
                    ex_std_data[key] = value
            
            ex_dist_data = {}
            ex_dist_path = CONST.CURRENT_PATH + '/' + stroke + '/ex_dist.csv'
            with open(ex_dist_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)

                for row in reader:
                    key = (int(row[CONST.TIMESTAMP]), row[CONST.FEATURE])
                    value = float(row[CONST.VALUE])
                    ex_dist_data[key] = value
            
            
            score = 0
            max = []
            for key in distance.keys():
                distsum = (distance[key] - ex_dist_data[key]) / ex_std_data[key]
                if distsum > 0:
                    score += distsum / len(distance)
                    max.append((key, distsum / len(distance)))
            
            self.score = score
            self.max = sorted(max, key=lambda x: x[1], reverse=True)

        except Exception as e:
            return e

    def interpret(self):
        try:
            try:
                self.score
            except AttributeError:
                raise ex.MethodOrderError
            
            res_list = [0] * (CONST.SWING_BEFORE_IMPACT + CONST.SWING_AFTER_IMPACT)
            for tuple in self.max:
                for i in range(CONST.SWING_WINDOW_SIZE):
                    lw = CONST.SWING_WINDOW_SIZE / (tuple[0][0] + i + 1)
                    if lw < 1: lw = 1
                    rw = CONST.SWING_WINDOW_SIZE / (CONST.SWING_BEFORE_IMPACT + CONST.SWING_AFTER_IMPACT - tuple[0][0] - i)
                    if rw < 1: rw = 1

                    res_list[tuple[0][0] + i] += tuple[1] * lw * rw
            
            self.res = res_list
            self.res_prepare = sum(res_list[:CONST.SWING_PREPARE_END]) / CONST.SWING_PREPARE_END
            self.res_impact = sum(res_list[CONST.SWING_PREPARE_END:CONST.SWING_IMPACT_END]) / (CONST.SWING_IMPACT_END - CONST.SWING_PREPARE_END)
            self.res_follow = sum(res_list[CONST.SWING_IMPACT_END:]) / (CONST.SWING_BEFORE_IMPACT + CONST.SWING_AFTER_IMPACT - CONST.SWING_IMPACT_END)

            # 최고 속도
            self.res_max = 0
            for index in range(CONST.SWING_PREPARE_END, CONST.SWING_IMPACT_END):
                max_vel = (self.raw_data.at[index, CONST.ACCX] ** 2 + self.raw_data.at[index, CONST.ACCY] ** 2 + self.raw_data.at[index, CONST.ACCZ] ** 2) ** (1/2)
                if max_vel > self.res_max:
                    self.res_max = max_vel
            

            self.feedback = []
            ex_avg_raw = pd.read_csv(CONST.CURRENT_PATH + '/' + self.stroke + '/ex_avg_raw.csv')

            # 준비 스윙
            cols_all = ex_avg_raw.columns.tolist()
            valres = {}
            stmres = {}
            for col in cols_all:
                valres[col] = 0
                stmres[col] = 0
                for index in range(CONST.SWING_PREPARE_END):
                    if ex_avg_raw.at[index, col] >= 0:
                        dif = ex_avg_raw.at[index, col] - self.raw_data.at[index, col]
                        if abs(dif) > abs(valres[col]):
                            valres[col] = dif
                            stmres[col] = index
                    else:
                        dif = self.raw_data.at[index, col] - ex_avg_raw.at[index, col]
                        if abs(dif) > abs(valres[col]):
                            valres[col] = dif
                            stmres[col] = index

            if (valres[CONST.ACCX] + valres[CONST.ACCY] + valres[CONST.ACCZ]) > 2:
                self.feedback.append('000') # 준비 스윙 속도 부족
                if (valres[CONST.ACCX] + valres[CONST.ACCY] + valres[CONST.ACCZ]) > 3:
                    self.feedback.pop()
                    self.feedback.append('001') # 준비 스윙 속도 부족
            else:
                self.feedback.append('500') # 백스윙 속도 적당

            if abs(valres[CONST.ACCX]) > 2 or abs(valres[CONST.ACCY]) > 2 or abs(valres[CONST.ACCZ]) > 2:
                self.feedback.append('010') # 준비 스윙 궤적 이상

                if abs(valres[CONST.RVP]) > 1000 or abs(valres[CONST.RVR]) > 1000:
                    self.feedback.append('020') # 010 준비 스윙 손목 방향 잘못됨
                    if abs(valres[CONST.RVP]) > 1000 or abs(valres[CONST.RVR]) > 1500:
                        self.feedback.pop()
                        self.feedback.append('021') # 010 준비 스윙 손목 방향 잘못됨

                elif abs(valres[CONST.RVY]) > 1000:
                    self.feedback.append('030') # 010 준비 스윙 깊이 부족
                    if abs(valres[CONST.RVY]) > 1500:
                        self.feedback.pop()
                        self.feedback.append('031') # 010 준비 스윙 깊이 부족
            else:
                self.feedback.append('510') # 백스윙 궤적 올바름
            
            # 임펙트
            valres = {}
            stmres = {}
            for col in cols_all:
                valres[col] = 0
                stmres[col] = 0
                for index in range(CONST.SWING_PREPARE_END, CONST.SWING_IMPACT_END):
                    if ex_avg_raw.at[index, col] >= 0:
                        dif = ex_avg_raw.at[index, col] - self.raw_data.at[index, col]
                        if abs(dif) > abs(valres[col]):
                            valres[col] = dif
                            stmres[col] = index
                    else:
                        dif = self.raw_data.at[index, col] - ex_avg_raw.at[index, col]
                        if abs(dif) > abs(valres[col]):
                            valres[col] = dif
                            stmres[col] = index

            if (valres[CONST.ACCX] + valres[CONST.ACCY] + valres[CONST.ACCZ]) > 3:
                self.feedback.append('100') # 임펙트 스윙 속도 부족
                if (valres[CONST.ACCX] + valres[CONST.ACCY] + valres[CONST.ACCZ]) > 5:
                    self.feedback.pop()
                    self.feedback.append('101') # 임펙트 스윙 속도 매우 부족
            else:
                self.feedback.append('400') #임펙트 스윙 속도 적당
                
            if abs(valres[CONST.RVP]) > 1000 or abs(valres[CONST.RVR]) > 1000 or abs(valres[CONST.RVY]) > 1000:
                self.feedback.append('110') # 임펙트 스윙 손목 방향 잘못됨

                if abs(valres[CONST.RVR]) > 1000:
                    self.feedback.append('120') # 110 임펙트 스윙 손목 회전 미사용

                elif abs(valres[CONST.RVP]) > 1000 or abs(valres[CONST.RVY]) > 1000:
                    self.feedback.append('130') # 110 임펙트 스윙 손목 회전 이상
                    if abs(valres[CONST.RVP]) > 1500 or abs(valres[CONST.RVY]) > 1500:
                        self.feedback.pop()
                        self.feedback.append('131') # 110 임펙트 스윙 손목 회전 이상
            else:
                self.feedback.append('610') #임펙트 스윙 손목 회전 감지
            
            # 팔로스로우
            valres = {}
            stmres = {}
            for col in cols_all:
                valres[col] = 0
                stmres[col] = 0
                for index in range(CONST.SWING_IMPACT_END, CONST.SWING_BEFORE_IMPACT + CONST.SWING_AFTER_IMPACT):
                    if ex_avg_raw.at[index, col] >= 0:
                        dif = ex_avg_raw.at[index, col] - self.raw_data.at[index, col]
                        if abs(dif) > abs(valres[col]):
                            valres[col] = dif
                            stmres[col] = index
                    else:
                        dif = self.raw_data.at[index, col] - ex_avg_raw.at[index, col]
                        if abs(dif) > abs(valres[col]):
                            valres[col] = dif
                            stmres[col] = index

            if abs(valres[CONST.ACCX]) > 6 or abs(valres[CONST.ACCY]) > 6 or abs(valres[CONST.ACCZ]) > 6:
                self.feedback.append('200') # 임펙트 이후 자세가 부자연스러움

                if abs(valres[CONST.ACCX]) > 6 or abs(valres[CONST.ACCY]) > 6 or abs(valres[CONST.ACCZ]) > 12:
                    self.feedback.pop()
                    self.feedback.append('201') # 임펙트 이후 자세가 부자연스러움
            else:
                self.feedback.append('700') # 임펙트 이후 자세 안정
                
            if valres[CONST.RVP] > 1000 and stmres[CONST.RVP] < 12:
                self.feedback.append('210') # 손목회전이 민첩하지 않음

                if valres[CONST.RVP] > 1500:
                    self.feedback.pop()
                    self.feedback.append('211') # 손목회전이 민첩하지 않음
            else:
                self.feedback.append('710') # 임펙트 이후 손목회전이 자연스러움

            if valres[CONST.RVR] > 1000:
                self.feedback.append('220') # 팔로우 스윙 과정에서 손목이 꺾임

                if valres[CONST.RVR] > 1500:
                    self.feedback.pop()
                    self.feedback.append('221') # 팔로우 스윙 과정에서 손목이 꺾임
            else:
                self.feedback.append('720') # 임펙트 이후 손목이 꺾이지 않음

        except Exception as e:
            return e