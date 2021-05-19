# 测试将采集到的数据（[250,8]）划分为带有时间片维度的数据（[n,8,50]）
import numpy as np
import os

from configReader import ConfigReader

CONF = ConfigReader()
ACTIONS = ["left", "right", "up", "down", "none"]
RESHAPE = (-1, CONF.channels_num, CONF.time_slot)

def load(starting_dir="raw_data", cfm = "default"):
    if cfm != "default":
        CONF.setMode(cfm)
    
    # actions ={"up"}
    all_data = {}
    train_data = {}
    test_data = {}
    validate_data = {}
    combined_train = []
    combined_test = []
    combined_validate = []

    for action in ACTIONS:
        if action not in all_data:
            all_data[action] = []
        if action not in train_data:
            train_data[action] = []
        if action not in test_data:
            test_data[action] = []
        if action not in validate_data:
            validate_data[action] = []

        data_dir = os.path.join(starting_dir,action)
        for item in os.listdir(data_dir):
            data = np.load(os.path.join(data_dir, item))
            data = data[40:-10]
            data = (data+100)/1e6
            print(data.shape)
            item=cutData(data)
            all_data[action].append(item)
        all_data[action]=np.array(all_data[action])
        np.random.shuffle(all_data[action])

        data_size = len(all_data[action])
        test_size = int(data_size*0.2)
        val_size = int(data_size*0.2)

        test_data[action], rest_data = np.split(all_data[action],[test_size])
        validate_data[action], train_data[action] = np.split(rest_data,[val_size])
        # print("split.",time.time()-begin_time)
        
        train_data[action] = train_data[action].reshape(RESHAPE)
        test_data[action] = test_data[action].reshape(RESHAPE)
        validate_data[action] = validate_data[action].reshape(RESHAPE)

        print(action, "test", test_data[action].shape)
        print(action, "validate", validate_data[action].shape)
        print(action, "train", train_data[action].shape)

        # act_1hot = np.zeros_like(actions, int)
        # act_1hot[int(np.argwhere(np.array(actions)==action))] = 1 #将训练数据写成[data, tag]的记录对，其中tag使用onehot表示
        # tag = act_1hot

        act_num = int(np.argwhere(np.array(ACTIONS)==action))
        tag = act_num

        for data in train_data[action]:
            combined_train.append([data, tag])
        
        for data in test_data[action]:
            combined_test.append([data, tag])
        
        for data in validate_data[action]:
            combined_validate.append([data, tag])

    np.random.shuffle(combined_train)
    np.random.shuffle(combined_test)
    np.random.shuffle(combined_validate)
    print("train:",len(combined_train))
    print("test:",len(combined_test))
    print("validate:",len(combined_validate))
    return combined_train, combined_test, combined_validate


def tag_divide(combined_data):
    data_X = []
    data_y = []
    for X, y in combined_data:
        data_X.append(X)
        data_y.append(y)
    data_X = np.array(data_X).reshape((-1, len(CONF.selected_channels), CONF.time_slot))
    data_y = np.array(data_y)
    return data_X, data_y
        

def cutData(raw_data,stride=CONF.data_stride,time_slot=CONF.time_slot):
    N=raw_data.shape[0]
    c=raw_data.shape[1]
    n=int((N-time_slot)/stride+1)
    # print(N, c, f, n)
    data=np.ones((n,time_slot,c), dtype="float16")
    for i in range(0,N,stride):
        # print(i+time_slot, raw_data.shape[0])
        if i+time_slot<=raw_data.shape[0]:
            data[int(i/stride)]=raw_data[np.arange(i,i+time_slot)]
    data=data.transpose(0,2,1)
    # print(data.shape)
    return data
