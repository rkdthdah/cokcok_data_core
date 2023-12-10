from scipy.signal import find_peaks
import pandas as pd
import math
import joblib
import itertools
import numpy as np

from . import exception as ex
from .constant import CONST
from . import analysis


class SwingClassification:
    """
    경기 데이터로부터 스윙 분류 class

    analysis의 SwingAnalysis class,
    constant, exception, util 모듈,
    feature.txt, knn_model.joblib 파일에 의존성을 갖는다

    field:
    data (DataFrame): 분류 대상 경기 data. class 생성자로 배정된다
    raw_data(DataFrame): 분류 후 스윙 분석에 사용하기 위한 data
    preproccessed_data (DataFrame): 전처리 된 분석 대상 data
    X_y (DataFrame): 분석된 data의 스윙 시작, 끝 index 정보 기록
    classification_result (String[]): data에서 찾고, 분류된 스윙 종류를 순서대로 기록
    swing_classification_analysis (SwingAnalysis[]): 찾은 스윙들의 스윙 분석 class들의 리스트

    method:
    __init__(df): 전역변수 data 배정
    preprocess(data) return data: data 전처리
    findPeak(df, window_size, acc_threshold, rr_threshold, feat_name, feat_threshold, peak_threshold) return res_peaks (int[]): 피크 찾기
    classification(): 경기 데이터로부터 stroke 추출 및 분류, 전역변수 X_y, classification_result 배정, 함수 preprocess, findpeak를 사용한다
    makeAnalysisData() return swing_classification_analysis: 스윙 분류 결과로부터 각 스윙 분석, 함수 classification에 후행한다
    """
    
    def __init__(self, df):
        self.data = df

    def preprocess(self, data):
        for r in range(len(data)):
            data.at[r, CONST.RVP] = data.at[r, CONST.RVP] * 180 / math.pi
            data.at[r, CONST.RVR] = data.at[r, CONST.RVR] * 180 / math.pi
            data.at[r, CONST.RVY] = data.at[r, CONST.RVY] * 180 / math.pi
        
        data.drop(CONST.PITCH, axis=1, inplace=True)
        data.drop(CONST.ROLL, axis=1, inplace=True)
        data.drop(CONST.YAW, axis=1, inplace=True)
        
        return data

    def findPeak(self, df, window_size, acc_threshold, rr_threshold, feat_name, feat_threshold, peak_threshold):
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

        return res_peaks
    
    def classification(self):
        try:
            self.raw_data = self.data.copy()
            preproccessed_data = self.preprocess(self.data)

            # make feature dataframe
            X_y = pd.DataFrame(columns=[CONST.CLASS_START, CONST.CLASS_END])

            peaks = self.findPeak(preproccessed_data, CONST.CLASS_WINDOW_SIZE, CONST.CLASS_ACC_THR, CONST.CLASS_RR_THR, CONST.ACCZ, CONST.CLASS_FEAT_THR, CONST.CLASS_PEAK_THR)

            for peak in peaks:
                d = pd.DataFrame({CONST.CLASS_START: [int(peak - CONST.CLASS_WINDOW_SIZE / 2)],
                                CONST.CLASS_END: [int(peak + CONST.CLASS_WINDOW_SIZE / 2)]})
                
                X_y = pd.concat([X_y, d], ignore_index=True)
            
            self.X_y = X_y

            # Add feature which depends only on one sensor, like range
            def add_feature(fname, sensor):
                v = [fname(preproccessed_data[int(row[CONST.CLASS_START]):int(row[CONST.CLASS_END])], sensor) for index, row in X_y.iterrows()]
                X_y[fname.__name__ + str(sensor)] = v
                
            # Add feature which depends on more than one sensors, like magnitude
            def add_feature_mult_sensor(fname, sensors):
                v = [fname(preproccessed_data[int(row[CONST.CLASS_START]):int(row[CONST.CLASS_END])], sensors) for index, row in X_y.iterrows()]
                
                name = "_".join(sensors)
                X_y[fname.__name__ + name] = v
            
            # Range 
            def range_(df, sensor):
                return np.max(df[sensor]) - np.min(df[sensor])
            for sensor in CONST.CLASS_COLS:
                add_feature(range_, sensor)

            # Minimum
            def min_(df, sensor):
                return np.min(df[sensor])
            for sensor in CONST.CLASS_COLS:
                add_feature(min_, sensor)

            # Maximum
            def max_(df, sensor):
                return np.max(df[sensor])
            for sensor in CONST.CLASS_COLS:
                add_feature(max_, sensor)

            # Average
            def avg_(df, sensor):
                return np.mean(df[sensor])
            for sensor in CONST.CLASS_COLS:
                add_feature(avg_, sensor)

            # Absolute Average
            def absavg_(df, sensor):
                return np.mean(np.absolute(df[sensor]))
            for sensor in CONST.CLASS_COLS:
                add_feature(absavg_, sensor)

            def kurtosis_f_(df , sensor):
                from scipy.stats import kurtosis 
                val = kurtosis(df[sensor],fisher = True)
                return val
            for sensor in CONST.CLASS_COLS:
                add_feature(kurtosis_f_, sensor)

            def kurtosis_p_(df , sensor):
                from scipy.stats import kurtosis 
                val = kurtosis(df[sensor],fisher = False)
                return val
            for sensor in CONST.CLASS_COLS:
                add_feature(kurtosis_p_, sensor)

            #skewness
            def skewness_statistic_(df, sensor):
                if(len(df) == 0):
                    print(df)
                from scipy.stats import skewtest 
                statistic, pvalue = skewtest(df[sensor], nan_policy='propagate')
                return statistic
            for sensor in CONST.CLASS_COLS:
                add_feature(skewness_statistic_, sensor)

            def skewness_pvalue_(df, sensor):
                from scipy.stats import skewtest 
                statistic, pvalue = skewtest(df[sensor])
                return pvalue
            for sensor in CONST.CLASS_COLS:
                add_feature(skewness_pvalue_, sensor)

            # Standard Deviation
            def std_(df, sensor):
                return np.std(df[sensor])
            for sensor in CONST.CLASS_COLS:
                add_feature(std_, sensor)

            #angle between two vectors
            def anglebetween_(df, sensors):
                v1 = sensors[0]
                v2 = sensors[1]
                v1_u = df[v1] / np.linalg.norm(df[v1])
                v2_u = df[v2] / np.linalg.norm(df[v2])
                return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
            for comb in list(itertools.combinations(CONST.CLASS_COLS, 2)):
                add_feature_mult_sensor(anglebetween_, comb)

            #inter quartile range
            def iqr_(df, sensor):
                from scipy import stats
                return stats.iqr(df[sensor])
            for sensor in CONST.CLASS_COLS:
                add_feature(iqr_, sensor)

            # Max position - min position (relative difference)
            def maxmin_relative_pos_(df, sensor):
                return np.argmax(np.array(df[sensor])) - np.argmin(np.array(df[sensor]))
            for sensor in CONST.CLASS_COLS:
                add_feature(maxmin_relative_pos_, sensor)
            
            X_y = X_y.dropna()


            # KNN predict
            # Read Features
            with open(CONST.CURRENT_PATH + '/features.txt') as f:
                features = f.read().strip().split("\n")
            f.close()

            model_filename = CONST.CURRENT_PATH + '/knn_model.joblib'
            loaded_knn_model = joblib.load(model_filename)

            predicted_labels = loaded_knn_model.predict(X_y[features].values)

            self.classification_result = predicted_labels
        
        except Exception as e:
            return e
    
    def makeAnalysisData(self):
        res_classes = []

        for i in range(len(self.classification_result)):

            df = self.raw_data.iloc[self.X_y.at[i, CONST.CLASS_START]:self.X_y.at[i, CONST.CLASS_END], :].copy()
            find = analysis.SwingAnalysis(df.reset_index(drop=True))
            find.analysis(self.classification_result[i], False)
            swing_score = max((3-find.score)*100/3,0)
            res_classes.append((self.classification_result[i],swing_score))

        self.swing_classification_analysis = res_classes

        return res_classes