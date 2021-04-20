# 测试将采集到的数据（[250,8,60]）划分为带有时间片维度的数据（[n,8,50,60]）
import numpy as np
import os

from tensorflow.python.keras.utils.generic_utils import default
from configReader import ConfigReader

CONF = ConfigReader()
RESHAPE = (-1, CONF.time_slot, CONF.frequency_slot, CONF.channels_num)

def load(starting_dir, cfm = "default"):
    if cfm != "default":
        CONF.setMode(cfm)
    
    actions = CONF.actions
    all_data = {}
    for action in actions:
        if action not in all_data:
            all_data[action] = []

        data_dir = os.path.join(starting_dir,action)
        for item in os.listdir(data_dir):
            #print(action, item)
            data = np.load(os.path.join(data_dir, item))
            # for item in data:
                # print(data.shape)
                # training_data[action].append(item)
            item=cutData(data)
            # print(item.shape)
            all_data[action].append(item)
        all_data[action]=np.array(all_data[action]).reshape(RESHAPE)
        all_data[action] = all_data[action][...,CONF.selected_channels]
        print(action,all_data[action].shape)

    lengths = [len(all_data[action]) for action in actions]
    if CONF.getAttr("default", "cut_flag") == "cut":
        for action in actions:
            np.random.shuffle(all_data[action])  # note that regular shuffle is GOOF af
            all_data[action] = all_data[action][:min(lengths)]        ## 规格化,将三个标签的长度截取成相同，保证训练样本中的比例为1:1:1 ###
            lengths = [len(all_data[action]) for action in actions]

    combined_data = []
    for action in actions:
        act1hot = np.zeros_like(actions, int)
        act1hot[int(np.argwhere(np.array(actions)==action))] = 1
        # print(act1hot)
        # act1hot = list(act1hot)
        # print(act1hot)
        for data in all_data[action]:
            combined_data.append([data, act1hot])        #将训练数据写成[data, tag]的记录对，其中tag使用onehot表示

    np.random.shuffle(combined_data)
    print("total length:",len(combined_data))
    return combined_data



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