import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import math
import os
import itertools
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Timestamp,AccelerationX,AccelerationY,AccelerationZ,Pitch,Yaw,Roll,RRX,RRY,RRZ
# backhand drive = bd, backhand hairpin = bn, backhand highclear = bh, backhand underclear = bu,
# forehand drive = fd, forehand drop = fp, forehand hairpin = fn, forehand highclear = fh, forehand smash = fs, forehand underclear = fu
# long service = ls, short service = ss

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

    return res_peaks

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


data_list = getData('stroke data/')
print(data_list)

window_size = 30
acc_threshold = 2
rr_threshold = 200
feat_name = 'AccelerationZ'
feat_threshold = 1
peak_threshold = 3

# make final dataframe
X_y = pd.DataFrame(columns=['StartFrame', 'EndFrame', 'PersonID', 'ShotName'])

for key, value in data_list.items():
    peaks = findPeak(value, window_size, acc_threshold, rr_threshold, feat_name, feat_threshold, peak_threshold)

    for peak in peaks:
        d = pd.DataFrame({'StartFrame': [int(peak - window_size/2)],
            'EndFrame': [int(peak + window_size/2)], 
            'PersonID': [key.split(" ")[0]], 
            'ShotName': [key.split(" ")[1]],
            'ShotNum' : [key.split(" ")[2]]})
        
        X_y = pd.concat([X_y, d], ignore_index=True)

# add features
features = []
cols = ['AccelerationX', 'AccelerationY', 'AccelerationZ', 'RRX', 'RRY', 'RRZ']
# Some helper functions

# Add feature which depends only on one sensor, like range
def add_feature(fname, sensor):
    v = [fname(data_list[str(row['PersonID']) + " " + str(row['ShotName']) + " " + str(row['ShotNum'])][int(row['StartFrame']):int(row['EndFrame'])],
               sensor) for index, row in X_y.iterrows()]
    X_y[fname.__name__ + str(sensor)] = v
    if(fname.__name__ + str(sensor) not in features):
        features.append(fname.__name__ + str(sensor))
    print("Added feature " + fname.__name__ + str(sensor) + " for " + str(len(v)) + " rows.")
    
# Add feature which depends on more than one sensors, like magnitude
def add_feature_mult_sensor(fname, sensors):
    v = [fname(data_list[str(row['PersonID']) + " " + str(row['ShotName']) + " " + str(row['ShotNum'])][int(row['StartFrame']):int(row['EndFrame'])],
               sensors) for index, row in X_y.iterrows()]
    
    name = "_".join(sensors)
    X_y[fname.__name__ + name] = v
    if(fname.__name__ + name not in features):
        features.append(fname.__name__ + name)
    print("Added feature " + fname.__name__ + name + " for " + str(len(v)) + " rows.")

# Range 
def range_(df, sensor):
    return np.max(df[sensor]) - np.min(df[sensor])
for sensor in cols:
    add_feature(range_, sensor)

# Minimum
def min_(df, sensor):
    return np.min(df[sensor])
for sensor in cols:
    add_feature(min_, sensor)

# Maximum
def max_(df, sensor):
    return np.max(df[sensor])
for sensor in cols:
    add_feature(max_, sensor)

# Average
def avg_(df, sensor):
    return np.mean(df[sensor])
for sensor in cols:
    add_feature(avg_, sensor)

# Absolute Average
def absavg_(df, sensor):
    return np.mean(np.absolute(df[sensor]))
for sensor in cols:
    add_feature(absavg_, sensor)

def kurtosis_f_(df , sensor):
    from scipy.stats import kurtosis 
    val = kurtosis(df[sensor],fisher = True)
    return val
for sensor in cols:
    add_feature(kurtosis_f_, sensor)

def kurtosis_p_(df , sensor):
    from scipy.stats import kurtosis 
    val = kurtosis(df[sensor],fisher = False)
    return val
for sensor in cols:
    add_feature(kurtosis_p_, sensor)

#skewness
def skewness_statistic_(df, sensor):
    if(len(df) == 0):
        print(df)
    from scipy.stats import skewtest 
    statistic, pvalue = skewtest(df[sensor], nan_policy='propagate')
    return statistic
for sensor in cols:
    add_feature(skewness_statistic_, sensor)

def skewness_pvalue_(df, sensor):
    from scipy.stats import skewtest 
    statistic, pvalue = skewtest(df[sensor])
    return pvalue
for sensor in cols:
    add_feature(skewness_pvalue_, sensor)

# #entropy 
# def entropy_(df, sensor):
#     from scipy.stats import entropy
#     ent = entropy(df[sensor])
#     return ent
# for sensor in cols:
#     add_feature(entropy_, sensor)

# Standard Deviation
def std_(df, sensor):
    return np.std(df[sensor])
for sensor in cols:
    add_feature(std_, sensor)

#angle between two vectors
def anglebetween_(df, sensors):
    v1 = sensors[0]
    v2 = sensors[1]
    v1_u = df[v1] / np.linalg.norm(df[v1])
    v2_u = df[v2] / np.linalg.norm(df[v2])
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
for comb in list(itertools.combinations(cols, 2)):
    add_feature_mult_sensor(anglebetween_, comb)

#inter quartile range
def iqr_(df, sensor):
    from scipy import stats
    return stats.iqr(df[sensor])
for sensor in cols:
    add_feature(iqr_, sensor)

# Max position - min position (relative difference)
def maxmin_relative_pos_(df, sensor):
    return np.argmax(np.array(df[sensor])) - np.argmin(np.array(df[sensor]))
for sensor in cols:
    add_feature(maxmin_relative_pos_, sensor)

script_dir = os.path.dirname(os.path.abspath(__file__))
# Save all the features in a txt file for later use.
with open(script_dir + '/stroke data/features.txt', 'w') as f:
    for feature in features:
        f.write("%s\n" %feature)
# Save X_y as csv file for using in (classical) ML models
X_y.to_csv(script_dir + '/X_y.csv', index=False) 



# Read Features 
with open(script_dir + '/stroke data/features.txt') as f:
    features = f.read().strip().split("\n")
f.close()

# Load data
X_y = pd.read_csv(script_dir + '/X_y.csv')
X_y = X_y.dropna()
shot_labels = X_y.ShotName.unique()

# Train Test split randomly
from sklearn.model_selection import train_test_split
train, test = train_test_split(X_y, test_size=0.2, random_state=42)
print(X_y)
print(X_y[features].values)
print(X_y["ShotName"].values)

print("Ssssssssssssssssssssssssss")


X_train = X_y[features].values
Y_train = X_y["ShotName"].values
X_test  = X_y[features].values
Y_test  = X_y["ShotName"].values

# X_train = train[features].values
# Y_train = train["ShotName"].values
# X_test  = test[features].values
# Y_test  = test["ShotName"].values
print(X_train)
print(Y_train)

# Helper function for plotting confusion matrix
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, shots,
                          model_name,
                          normalize=False,
                          cmap=plt.cm.Wistia):
    tick_marks = np.arange(len(shots))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.yticks(tick_marks, shots)
    plt.title("Confusion matrix - " + model_name)
    plt.colorbar()
    plt.xticks(tick_marks, shots, rotation='vertical')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black")
    plt.tight_layout()
    plt.ylabel('True Shot')
    plt.xlabel('Predicted Shot')
    plt.savefig(script_dir + '/plots/' + 'Confusion matrix - ' + model_name)


# Logistic Regression
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# # Save the hyperparameters ie C value and loss-function type:
# parameters = {'C':[0.01,0.1,1,10,20,30], 'penalty':['l2']}
# log_reg_clf = linear_model.LogisticRegression(max_iter=1000)
# log_reg_model = GridSearchCV(log_reg_clf, param_grid=parameters, cv=3,verbose=1, n_jobs=8)

# log_reg_model.fit(X_train,Y_train)
# y_pred = log_reg_model.predict(X_test)
# # y_prob = log_reg_model.predict_proba(X_test)
# # print(y_prob)
# accuracy = metrics.accuracy_score(y_true=Y_test,y_pred=y_pred)

# # Accuracy of our stroke detectiony
# print('Accuracy of strokes detection:   {}\n\n'.format(accuracy))
     
# # confusion matrix
# cm = metrics.confusion_matrix(Y_test, y_pred)

# # plot confusion matrix
# plt.figure(figsize=(8,8))
# plt.grid(False)
# plot_confusion_matrix(cm, model_name="Logistic Regression", 
#                       shots=shot_labels, normalize=True)
# plt.show()
    
# # get classification report
# print("Classifiction Report for this model")
# classification_report = metrics.classification_report(Y_test, y_pred)
# print(classification_report)


# KNN
from sklearn.neighbors import KNeighborsClassifier
import joblib

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, Y_train)
model_filename = script_dir + '/knn_model.joblib'
joblib.dump(knn, model_filename)
y_pred = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_true=Y_test,y_pred=y_pred)

# Accuracy of our stroke detectiony
print('Accuracy of strokes detection:   {}\n\n'.format(accuracy))
     
# confusion matrix
cm = metrics.confusion_matrix(Y_test, y_pred)
    
# plot confusion matrix
plt.figure(figsize=(8,8))
plt.grid(False)
plot_confusion_matrix(cm, model_name='KNeighborsClassifier',
                      shots=shot_labels, normalize=True, )
plt.show()
    
# get classification report
print("Classifiction Report for this model")
classification_report = metrics.classification_report(Y_test, y_pred)
print(classification_report)


# # Linear SVC
# from sklearn.svm import LinearSVC
# parameters = {'C':[0.125, 0.5, 1, 2, 8, 16]}
# lr_svc_reg_clf = LinearSVC(tol=0.00005, max_iter=1000)
# lr_svc_reg_model = GridSearchCV(lr_svc_reg_clf, param_grid=parameters, n_jobs=8, verbose=1)

# lr_svc_reg_model.fit(X_train,Y_train)
# y_pred = lr_svc_reg_model.predict(X_test)
# accuracy = metrics.accuracy_score(y_true=Y_test,y_pred=y_pred)
# # Accuracy of our stroke detectiony
# print('Accuracy of strokes detection:   {}\n\n'.format(accuracy))
     
# # confusion matrix
# cm = metrics.confusion_matrix(Y_test, y_pred)
        
# # plot confusion matrix
# plt.figure(figsize=(8,8))
# plt.grid(False)
# plot_confusion_matrix(cm, model_name='LinearSVC',
#                       shots=shot_labels, normalize=True)
# plt.show()
    
# # get classification report
# print("Classifiction Report for this model")
# classification_report = metrics.classification_report(Y_test, y_pred)
# print(classification_report)


# # SVC with RBF kernel
# from sklearn.svm import SVC
# parameters = {'C':[2,8,16],\
#               'gamma': [ 0.0078125, 0.125, 2]}
# rbf_svm_clf = SVC(kernel='rbf')
# rbf_svm_model = GridSearchCV(rbf_svm_clf,param_grid=parameters,n_jobs=8)

# rbf_svm_model.fit(X_train,Y_train )
# y_pred = rbf_svm_model.predict(X_test)
# accuracy = metrics.accuracy_score(y_true=Y_test,y_pred=y_pred)
# # Accuracy of our stroke detectiony
# print('Accuracy of strokes detection:   {}\n\n'.format(accuracy))
     
# # confusion matrix
# cm = metrics.confusion_matrix(Y_test, y_pred)

# # plot confusion matrix
# plt.figure(figsize=(8,8))
# plt.grid(False)
# plot_confusion_matrix(cm, model_name='SVC', shots=shot_labels, normalize=True)
# plt.show()
    
# # get classification report
# print("Classifiction Report for this model")
# classification_report = metrics.classification_report(Y_test, y_pred)
# print(classification_report)


# # Dicision Tree
# from sklearn.tree import DecisionTreeClassifier
# parameters = {'max_depth':np.arange(3,20,2)}
# decision_trees_clf = DecisionTreeClassifier()
# decision_trees = GridSearchCV(decision_trees_clf, param_grid=parameters, n_jobs=8)

# decision_trees.fit(X_train,Y_train )
# y_pred = decision_trees.predict(X_test)
# accuracy = metrics.accuracy_score(y_true=Y_test,y_pred=y_pred)
# # Accuracy of our stroke detection
# print('Accuracy of strokes detection:   {}\n\n'.format(accuracy))
     
# # confusion matrix
# cm = metrics.confusion_matrix(Y_test, y_pred)

# # plot confusion matrix
# plt.figure(figsize=(8,8))
# plt.grid(False)
# plot_confusion_matrix(cm, model_name='Decision Tree',
#                       shots=shot_labels, normalize=True)
# plt.show()
    
# # get classification report
# print("Classifiction Report for this model")
# classification_report = metrics.classification_report(Y_test, y_pred)
# print(classification_report)


# # Random Forest
# from sklearn.ensemble import RandomForestClassifier
# params = {'n_estimators': np.arange(10,120,20), 'max_depth':np.arange(3,15,2)}
# rfclassifier_clf = RandomForestClassifier()
# rfclassifier = GridSearchCV(rfclassifier_clf, param_grid=params, n_jobs=8)

# rfclassifier.fit(X_train,Y_train )
# y_pred = rfclassifier.predict(X_test)
# accuracy = metrics.accuracy_score(y_true=Y_test,y_pred=y_pred)
# # Accuracy of our stroke detection
# print('Accuracy of strokes detection:   {}\n\n'.format(accuracy))
     
# # confusion matrix
# cm = metrics.confusion_matrix(Y_test, y_pred)

# # plot confusion matrix
# plt.figure(figsize=(8,8))
# plt.grid(False)
# plot_confusion_matrix(cm, model_name='Random Forest',
#                       shots=shot_labels, normalize=True)
# plt.show()

# # get classification report
# print("Classifiction Report for this model")
# classification_report = metrics.classification_report(Y_test, y_pred)
# print(classification_report)


# # Gradient Boosting
# from sklearn.ensemble import GradientBoostingClassifier
# param_grid = {'max_depth': np.arange(1,30,4), \
#              'n_estimators':np.arange(1,300,15)}
# gbdt_clf = GradientBoostingClassifier()
# gbdt_model = GridSearchCV(gbdt_clf, param_grid=param_grid, n_jobs=8)

# gbdt_model.fit(X_train,Y_train )
# y_pred = gbdt_model.predict(X_test)
# accuracy = metrics.accuracy_score(y_true=Y_test,y_pred=y_pred)
# # Accuracy of our stroke detectiony
# print('Accuracy of strokes detection:   {}\n\n'.format(accuracy))
     
# # confusion matrix
# cm = metrics.confusion_matrix(Y_test, y_pred)

# # plot confusion matrix
# plt.figure(figsize=(8,8))
# plt.grid(False)
# plot_confusion_matrix(cm, model_name='GradientBoostingClassifier',
#                       shots=shot_labels, normalize=True)
# plt.show()
    
# # get classification report
# print("Classifiction Report for this model")
# classification_report = metrics.classification_report(Y_test, y_pred)
# print(classification_report)



# # LSTM
# # Importing tensorflow
# np.random.seed(42)
# import tensorflow as tf
# tf.random.set_seed(42)

# from sklearn.preprocessing import StandardScaler
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.layers import Dropout
# from keras.callbacks import EarlyStopping
# from keras.layers import Conv1D
# from keras.layers import MaxPooling1D
# from keras.models import Sequential
# from keras.layers import LSTM
# from keras.layers import Dense, Dropout

# # ShotNames are the class labels
# ShotNames = {
#     'bd': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     'bh': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     'bn': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     'bu': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     'fd': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#     'fh': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#     'fn': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#     'fp': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#     'fs': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#     'fu': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#     'ls': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#     'ss': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
# }

# X = []
# y = []
# for index, row in X_y.iterrows():
#     df = data_list[str(row["PersonID"]) + " " + str(row["ShotName"]) + " 0" + str(row['ShotNum'])][row["StartFrame"]:row["EndFrame"]][cols]
#     X.append(df.to_numpy())
#     y.append(row["ShotName"])
# X = np.array(X)
# # One Hot Encoding
# y = np.array([ShotNames[i] for i in y])

# n_classes = len(ShotNames)
# timesteps = len(X[0])    # Window size
# input_dim = len(X[0][0]) # num of sensors = 6

# print(timesteps)
# print(input_dim)

# # Initializing parameters
# epochs = 200
# batch_size = 32
# n_hidden = 128

# # Loading the train and test data
# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train , Y_test = train_test_split(X, y, test_size=0.2)


# # Initiliazing the sequential model
# model = Sequential()
# # Configuring the parameters
# model.add(LSTM(n_hidden, input_shape=(timesteps, input_dim), return_sequences=True))
# # Adding a dropout layer
# model.add(Dropout(0.4))

# model.add(LSTM(int(n_hidden / 2)))
# model.add(Dropout(0.4))

# # Adding a dense output layer with sigmoid activation
# model.add(Dense(n_classes, activation='sigmoid'))

# # Adding Early Stopping callback
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# model.summary()

# # Compiling the model
# model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

# # Training the model
# model.fit(X_train, Y_train,
#           batch_size=batch_size,
#           validation_data=(X_test, Y_test),
#           epochs=epochs,
#           callbacks=[early_stopping])