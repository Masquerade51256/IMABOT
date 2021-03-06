# 测试将采集到的数据（[250,8,60]）划分为带有时间片维度的数据（[n,8,50,60]）
import numpy as np
import os
import time

from tensorflow.python.keras.utils.generic_utils import default
from configReader import ConfigReader

CONF = ConfigReader()
RESHAPE = (-1, CONF.time_slot, CONF.frequency_slot, CONF.channels_num)
ACTIONS = ["left", "right", "up", "down", "none"]

def load(starting_dir, cfm = "default"):
    if cfm != "default":
        CONF.setMode(cfm)
    
    # actions ={"up"}
    actions = ACTIONS
    all_data = {}
    train_data = {}
    test_data = {}
    validate_data = {}
    combined_train = []
    combined_test = []
    combined_validate = []

    for action in actions:
        if action not in all_data:
            all_data[action] = []
        if action not in train_data:
            train_data[action] = []
        if action not in test_data:
            test_data[action] = []
        if action not in validate_data:
            validate_data[action] = []

        data_dir = os.path.join(starting_dir,action)
        # print("items loading...")
        # begin_time = time.time()
        for item in os.listdir(data_dir):
            #print(action, item)
            data = np.load(os.path.join(data_dir, item))
            # for item in data:
                # print(data.shape)
                # training_data[action].append(item)
            item=cutData(data)
            # print(item.shape)
            all_data[action].append(item)
        # print("items loaded.",time.time()-begin_time)

        # print("np_array making...")
        # begin_time = time.time()
        all_data[action]=np.array(all_data[action])
        # print("np_array made.",time.time()-begin_time)

        # print("shuffling...")
        # begin_time = time.time()
        np.random.shuffle(all_data[action])
        # print("shuffled.",time.time()-begin_time)

        data_size = len(all_data[action])
        test_size = int(data_size*0.2)
        val_size = int(data_size*0.2)

        # print("spliting...")
        # begin_time = time.time()
        test_data[action], rest_data = np.split(all_data[action],[test_size])
        validate_data[action], train_data[action] = np.split(rest_data,[val_size])
        # print("split.",time.time()-begin_time)
        
        train_data[action] = train_data[action].reshape(RESHAPE)
        test_data[action] = test_data[action].reshape(RESHAPE)
        validate_data[action] = validate_data[action].reshape(RESHAPE)

        train_data[action] = train_data[action][...,CONF.selected_channels]
        test_data[action] = test_data[action][...,CONF.selected_channels]
        validate_data[action] = validate_data[action][...,CONF.selected_channels]

        print(action, "test", test_data[action].shape)
        print(action, "validate", validate_data[action].shape)
        print(action, "train", train_data[action].shape)

        act1hot = np.zeros_like(actions, int)
        act1hot[int(np.argwhere(np.array(actions)==action))] = 1 #将训练数据写成[data, tag]的记录对，其中tag使用onehot表示
        
        for data in train_data[action]:
            combined_train.append([data, act1hot])
        
        for data in test_data[action]:
            combined_test.append([data, act1hot])
        
        for data in validate_data[action]:
            combined_validate.append([data, act1hot])

    np.random.shuffle(combined_train)
    np.random.shuffle(combined_test)
    np.random.shuffle(combined_validate)
    print("train:",len(combined_train))
    print("test:",len(combined_test))
    print("validate:",len(combined_validate))
    return combined_train, combined_test, combined_validate



# def load_and_format(starting_dir,
#                     tag_flag = "onehot",
#                     cfm = "default"):
#     combined_data = load(starting_dir,tag_flag,cfm)
#     data_X = []
#     data_y = []
#     for X, y in combined_data:
#         data_X.append(X)
#         data_y.append(y)
#     return data_X, data_y

def tag_divide(combined_data):
    data_X = []
    data_y = []
    for X, y in combined_data:
        data_X.append(X)
        data_y.append(y)
    data_X = np.array(data_X, "float16").reshape((-1, CONF.time_slot, CONF.frequency_slot, len(CONF.selected_channels)))
    data_y = np.array(data_y)
    return data_X, data_y
        

def cutData(raw_data,stride=CONF.data_stride,time_slot=CONF.time_slot):
    """
    将原始数据划分为带有时间片维度的四维数据，即模型输入格式
    
    raw_data: 原始数据，要求三维数据，分别表示数据量、通道数和频率,
    其中N要大于等于time_slot
    """
    N=raw_data.shape[0]
    c=raw_data.shape[1]
    f=raw_data.shape[2]
    n=int((N-time_slot)/stride+1)
    # print(N, c, f, n)
    data=np.ones((n,time_slot,c,f), dtype="float16")
    for i in range(0,N,stride):
        # print(i+time_slot, raw_data.shape[0])
        if i+time_slot<=raw_data.shape[0]:
            data[int(i/stride)]=raw_data[np.arange(i,i+time_slot)]
    data=data.transpose(0,1,3,2)
    # print(data.dtype)
    return data

if __name__=="__main__":
    load("new_data")
    pass