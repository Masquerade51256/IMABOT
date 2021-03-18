import numpy as np
# 测试将采集到的数据（[250,8,60]）划分为带有时间片维度的数据（[n,8,50,60]）

all_num=12
single_num=6  #单次采集的数据量
channels_num=8
time_slot=2
freq_slot=6
raw_data = np.random.randn(all_num,channels_num,freq_slot)
print(raw_data.shape[0])
def format(raw_data,stride=2,single_num=6,channels_num=8,time_slot=2,freq_slot=6):
    data=np.ones((20,time_slot,channels_num,freq_slot))
    # data=[]
    for i in range(0,raw_data.shape[0],stride):
        if i+time_slot<raw_data.shape[0]:
            data[i]=raw_data[np.arange(i,i+time_slot)]
    data=data.transpose(0,2,1,3)
    print(data.shape)
    return data

print(format(raw_data))