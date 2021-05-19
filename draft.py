# import matplotlib.pyplot as plt
import numpy as np
# from time import sleep

# file = "horizontal_data/left/1618882899.npy"
# data = np.load(file)
# data= data.transpose(0,2,1)
# # for item in data:
# #     plt.plot(item)
# #     plt.show(block=True)

# item = data[0]
# print(item.shape)
# plt.plot(item)
# plt.show()










# Authors: Martin Billinger <martin.billinger@tugraz.at>
#
# License: BSD (3-clause)

# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn.pipeline import Pipeline
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.model_selection import ShuffleSplit, cross_val_score

# from mne import Epochs, pick_types, events_from_annotations
# from mne.channels import make_standard_montage
# from mne.io import concatenate_raws, read_raw_edf
# from mne.datasets import eegbci
# from mne.decoding import CSP

# print(__doc__)

# # #############################################################################
# # # Set parameters and read data

# # avoid classification of evoked responses by using epochs that start 1s after
# # cue onset.
# tmin, tmax = -1., 4.
# event_id = dict(hands=2, feet=3)
# subject = 1
# runs = [6, 10, 14]  # motor imagery: hands vs feet

# raw_fnames = eegbci.load_data(subject, runs)
# raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
# eegbci.standardize(raw)  # set channel names
# montage = make_standard_montage('standard_1005')
# raw.set_montage(montage)

# # strip channel names of "." characters
# raw.rename_channels(lambda x: x.strip('.'))

# # Apply band-pass filter
# raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

# events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

# picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
#                    exclude='bads')

# # Read epochs (train will be done only between 1 and 2s)
# # Testing will be done with a running classifier
# epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
#                 baseline=None, preload=True)
# epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
# print("=====epochs",epochs)
# print("=====epochs.events shape",epochs.events.shape)
# print("=====epochs.events",epochs.events)
# labels = epochs.events[:, -1] - 2

# # Define a monte-carlo cross-validation generator (reduce variance):
# scores = []
# epochs_data = epochs.get_data()
# epochs_data_train = epochs_train.get_data()
# cv = ShuffleSplit(10, test_size=0.2, random_state=42)
# cv_split = cv.split(epochs_data_train)

# # Assemble a classifier
# lda = LinearDiscriminantAnalysis()
# csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
# # n_component=4表示每个样本经过CSP输出的特征向量有四个维度，尚未发现该值与输入数据有何联系

# # Use scikit-learn Pipeline with cross_val_score function
# clf = Pipeline([('CSP', csp), ('LDA', lda)])
# print("=====clf cvs train data shape", epochs_data_train.shape)
# scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)

# # Printing the results
# class_balance = np.mean(labels == labels[0])
# class_balance = max(class_balance, 1. - class_balance) # 样本占比
# print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
#                                                           class_balance))

# print("=====csp fit transform input X shape", epochs_data.shape)
# print("=====csp fit transform input y shape", labels.shape)
# # plot CSP patterns estimated on full data for visualization
# csp_X = csp.fit_transform(epochs_data, labels)
# print("=====csp fit transform output shape", csp_X.shape)
# print("=====csp fit transform output", csp_X)










# import load_rawdata as load
# import numpy as np

# raw_data = np.load("raw_data/left/1620206849.npy")
# print(raw_data.shape)
# cut_data = load.cutData(raw_data)
# print(cut_data.shape)









# from mne.decoding import CSP
# import numpy as np
# import load_rawdata
# import tensorflow.keras as keras
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
# import os
# import time



# def csp_trans(X,y):
#     csp = CSP(n_components=4)
#     print("===============================debug===============================")
#     print("csp_train.py, fit_transform, X")
#     print("X shape", X.shape)
#     print("y shape", y.shape)
#     print("===============================debug===============================")
#     csp_X = csp.fit_transform(X=X,y=y)
#     print("csp_X shape", csp_X.shape)
#     print("csp_X", csp_X)
#     return csp_X

# def onehot_trans(y):
#     onehot_y = []
#     for i in y:
#         onehot = np.zeros((5,))
#         onehot[i] = 1
#         onehot_y.append(onehot)
#     onehot_y = np.array(onehot_y)
#     # print(onehot_y)
#     return onehot_y

# def cutData(raw_data,stride=20,time_slot=50):
#     N=raw_data.shape[0]
#     c=raw_data.shape[1]
#     n=int((N-time_slot)/stride+1)
#     # print(N, c, f, n)
#     data=np.ones((n,time_slot,c), dtype="float16")
#     for i in range(0,N,stride):
#         # print(i+time_slot, raw_data.shape[0])
#         if i+time_slot<=raw_data.shape[0]:
#             data[int(i/stride)]=raw_data[np.arange(i,i+time_slot)]
#     data=data.transpose(0,2,1)
#     # print(data.shape)
#     return data

# def tag_divide(combined_data):
#     data_X = []
#     data_y = []
#     for X, y in combined_data:
#         data_X.append(X)
#         data_y.append(y) 
#     data_X = np.array(data_X)
#     data_y = np.array(data_y)
#     print("data_X.shape", data_X.shape)
#     return data_X, data_y

# ACTIONS = ["left", "right", "up", "down", "none"]
# n, c, t = 0, 0, 0
# all_data = {}
# train_data = {}
# test_data = {}
# validate_data = {}
# combined_train = []
# combined_test = []
# combined_validate = []

# for action in ACTIONS:
#     if action not in all_data:
#         all_data[action] = []
#     if action not in train_data:
#         train_data[action] = []
#     if action not in test_data:
#         test_data[action] = []
#     if action not in validate_data:
#         validate_data[action] = []

#     data_dir = os.path.join("all_data",action)
#     # for item in os.listdir(data_dir):
#     for i in range(60):
#         item = os.listdir(data_dir)[i]
#         data = np.load(os.path.join(data_dir, item))
#         data = (data+100)/1e6
#         n = data.shape[0]
#         data = data.reshape(n,-1)
#         data=cutData(data)
#         _, c, t = data.shape
#         all_data[action].append(data)
#     all_data[action]=np.array(all_data[action])
#     np.random.shuffle(all_data[action])

#     data_size = len(all_data[action])
#     test_size = int(data_size*0.2)
#     val_size = int(data_size*0.2)

#     test_data[action], rest_data = np.split(all_data[action],[test_size])
#     validate_data[action], train_data[action] = np.split(rest_data,[val_size])
#     # print("split.",time.time()-begin_time)

#     reshape = (-1, c, t)

#     train_data[action] = train_data[action].reshape(reshape)
#     test_data[action] = test_data[action].reshape(reshape)
#     validate_data[action] = validate_data[action].reshape(reshape)

#     print(action, "test", test_data[action].shape)
#     print(action, "validate", validate_data[action].shape)
#     print(action, "train", train_data[action].shape)

#     # act_1hot = np.zeros_like(actions, int)
#     # act_1hot[int(np.argwhere(np.array(actions)==action))] = 1 #将训练数据写成[data, tag]的记录对，其中tag使用onehot表示
#     # tag = act_1hot

#     act_num = int(np.argwhere(np.array(ACTIONS)==action))
#     tag = act_num

#     for data in train_data[action]:
#         combined_train.append([data, tag])
        
#     for data in test_data[action]:
#         combined_test.append([data, tag])
        
#     for data in validate_data[action]:
#         combined_validate.append([data, tag])

# np.random.shuffle(combined_train)
# np.random.shuffle(combined_test)
# np.random.shuffle(combined_validate)
# print("train:",len(combined_train))
# print("test:",len(combined_test))
# print("validate:",len(combined_validate))

# train_X, train_y = tag_divide(combined_train)
# test_X, test_y = tag_divide(combined_test)
# val_X, val_y = tag_divide(combined_validate)


# train_X = csp_trans(train_X,train_y)
# test_X = csp_trans(test_X,test_y)
# val_X = csp_trans(val_X,val_y)

# train_y = onehot_trans(train_y)
# test_y = onehot_trans(test_y)
# val_y = onehot_trans(val_y)

# model = keras.Sequential()
# model.add(Dense(10, input_shape=train_X.shape[1:], activation='relu'))   
# model.add(Dense(20, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(5, activation='softmax'))
# model.summary()
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(train_X, train_y, batch_size=32, epochs=20, validation_data=(val_X, val_y))
# score = model.evaluate(test_X, test_y, batch_size=32)

# MODEL_NAME = f"csp_models/{round(score[1]*100,2)}acc-{round(score[0],2)}loss-{int(time.time())}.model"
# model.save(MODEL_NAME)









import control
import os
import load_rawdata
from mne.decoding import CSP

ACTIONS = control.ACTIONS
MODEL = control.MODEL
csp = CSP(n_components=4, reg=None, log=False, norm_trace=False)
csp_name = "1621049839"
csp.filters_ = np.load(os.path.join("csp_estimator",csp_name,"filters.npy"))
csp.patterns_ = np.load(os.path.join("csp_estimator",csp_name,"patterns.npy"))
csp.mean_ = np.load(os.path.join("csp_estimator",csp_name,"mean.npy"))
csp.std_ = np.load(os.path.join("csp_estimator",csp_name,"std.npy"))
data_path = "raw_data"
raw_data = np.load(os.path.join(data_path, "down", "1620799321.npy"))
raw_data = raw_data[:49]
head = 0
while(head+control.CONTROL_SLOT<len(raw_data)):
    data = raw_data[head:head+control.CONTROL_SLOT]
    print("data shape", data.shape)
    head+=1
    cut_data = load_rawdata.cutData(data, stride=1)
    print("cut data shape", cut_data.shape)
    input_data = csp.transform(cut_data)
    print(input_data.shape)

    _, c = input_data.shape
    input_data = input_data.reshape(-1,c,1)
    outputs = MODEL.predict(input_data)
    print(outputs.shape)
    act = control.decide(outputs)
    print(act)