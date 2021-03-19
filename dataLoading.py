import numpy as np
import os
import matplotlib.pyplot as plt
# 测试将采集到的数据（[250,8,60]）划分为带有时间片维度的数据（[n,8,50,60]）


def load_and_format(starting_dir="data",
                    stride=10,
                    time_slot=60,
                    actions = ["left", "right", "none"]):
    training_data = {}
    for action in actions:
        if action not in training_data:
            training_data[action] = []

        data_dir = os.path.join(starting_dir,action)
        for item in os.listdir(data_dir):
            #print(action, item)
            data = np.load(os.path.join(data_dir, item))
            # for item in data:
                # print(data.shape)
                # training_data[action].append(item)
            item=format(data,stride=stride,time_slot=time_slot)
            # print(item.shape)
            training_data[action].append(item)
        training_data[action]=np.array(training_data[action]).reshape(-1,60,60,8)
        print(action,training_data[action].shape)

    lengths = [len(training_data[action]) for action in actions]
    print(lengths)
    for action in actions:
        np.random.shuffle(training_data[action])  # note that regular shuffle is GOOF af
        training_data[action] = training_data[action][:min(lengths)]        ## 规格化,将三个标签的长度截取成相同，保证训练样本中的比例为1:1:1 ###

    lengths = [len(training_data[action]) for action in actions]
    print(lengths)

    combined_data = []
    for action in actions:
        for data in training_data[action]:

            if action == "left":
                combined_data.append([data, [1, 0, 0]])        #将训练数据写成[data, tag]的记录对，其中tag使用onehot表示

            elif action == "right":
                combined_data.append([data, [0, 0, 1]])

            elif action == "none":
                combined_data.append([data, [0, 1, 0]])

    np.random.shuffle(combined_data)
    print("length:",len(combined_data))

    return combined_data
        

def format(raw_data,stride=1,time_slot=2):
    """
    raw_data: 原始数据，要求三维数据，分别表示数据量、通道数和频率

    stride:
    
    time_slot:
    """
    N=raw_data.shape[0]
    c=raw_data.shape[1]
    f=raw_data.shape[2]
    n=int((N-time_slot)/stride+1)
    # print(n)
    data=np.ones((n,time_slot,c,f))
    # data=[]
    for i in range(0,N,stride):
        # print(i)
        if i+time_slot<raw_data.shape[0]:
            data[int(i/stride)]=raw_data[np.arange(i,i+time_slot)]
    data=data.transpose(0,1,3,2)
    # print(data.shape)
    return data

if __name__=="__main__":
    load_and_format()

    # all_num=12
    # channels_num=8
    # freq_slot=6
    # raw_data = np.random.randn(all_num,channels_num,freq_slot)
    # print(format(raw_data))