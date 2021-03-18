import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, Conv2D
from tensorflow.keras.layers import MaxPooling1D, MaxPooling2D, BatchNormalization
import os
import random
import time
import sys


ACTIONS = ["left", "right", "none"]
# reshape = (-1, 8, 60)
# reshape = (-1, 50, 60, 8)  
# 用于后续规格化，
# -1表示样本数量或批数，
# 500为时间区间0~50ms，（不确定此处单位是否为ms，需验证数据采集代码），后划分数据时应以此为参照
# 60为频率区间0~60hz，
# 8为通道数，由数据采集设备规格决定，
# （删除）由于后续keras中卷积层默认通道数在最后一个维度上，即channels_last，故此处需要将8放在最后
reshape = (-1, 8, 50, 60)   
### 将通道数放在第一位（除样本数量或批数），以适应采集数据的格式，
# 后续需要显式地将卷积层的data_format参数声明为channels_first

def create_data(starting_dir="data"):
    training_data = {}
    for action in ACTIONS:
        if action not in training_data:
            training_data[action] = []

        data_dir = os.path.join(starting_dir,action)
        for item in os.listdir(data_dir):
            #print(action, item)
            data = np.load(os.path.join(data_dir, item))
            for item in data:
                training_data[action].append(item)

    lengths = [len(training_data[action]) for action in ACTIONS]
    print(lengths)

    for action in ACTIONS:
        np.random.shuffle(training_data[action])  # note that regular shuffle is GOOF af
        training_data[action] = training_data[action][:min(lengths)]        ## 规格化,将三个标签的长度截取成相同，保证训练样本中的比例为1:1:1 ###

    lengths = [len(training_data[action]) for action in ACTIONS]
    print(lengths)
    
    # creating X, y 
    combined_data = []
    for action in ACTIONS:
        for data in training_data[action]:

            if action == "left":
                combined_data.append([data, [1, 0, 0]])        #one hot形式表示样本标签

            elif action == "right":
                #np.append(combined_data, np.array([data, [1, 0]]))
                combined_data.append([data, [0, 0, 1]])

            elif action == "none":
                combined_data.append([data, [0, 1, 0]])

    np.random.shuffle(combined_data)
    print("length:",len(combined_data))
    return combined_data


print("creating training data")
traindata = create_data(starting_dir="data")
train_X = []
train_y = []
for X, y in traindata:      ### X为训练集样本记录，即combine_data[]中的data；y为标记，即combine_data[]中用于表示“left”的[1,0,0]等 ###
    train_X.append(X)       ### shape = [11250(250*15*3), 8, 60]
    train_y.append(y)       ### shape = [11250, 3]

print("creating testing data")
testdata = create_data(starting_dir="validation_data")
test_X = []
test_y = []
for X, y in testdata:
    test_X.append(X)
    test_y.append(y)

print(len(train_X))
print(len(test_X))


print(np.array(train_X).shape)
train_X = np.array(train_X).reshape(reshape)
test_X = np.array(test_X).reshape(reshape)

train_y = np.array(train_y)
test_y = np.array(test_y)

### 选择模型 ###
model = Sequential()

### 构建网络 ###
# model.add(Conv1D(64, (3), input_shape=train_X.shape[1:]))       ### 输入层 input_shape = [8, 60]
# 上面一行为模型增加了一个Conv1D作为第一个层，即输入层
# Conv1D的参数中，filter=64，表示卷积核的数量，同时也是输出特征图的通道数  
# kernel_size=(3)，代表卷积核的大小
# 由于为输入层，所以还要提供input_shape参数，[1:]表示取该元组的第二（标号为1）到最后一个元素，1代表批的形状是一维的
model.add(Conv2D(64, (3,3), input_shape=train_X.shape[1:], data_format='channels_first'))   
### 尝试改为2维卷积核，因为输入数据是二维的（不包括批及通道维度）， 
# input_shape=(50,60,8)(实际输入时会增加批维度，即(batch_size,50,60,8))
# 根据卷积前后形状变化公式：HO=(HI+2padding-HK)/stride+1
# output_shape=(batch_size,48,58,64)
# 若想使卷积不缩小数据规模，即HO=HI，求解得padding=(h-1)/2，此处即padding=1

model.add(Activation('relu'))       ### 激发函数

# model.add(Conv1D(64, (2)))      ### Hiden_Layer1
model.add(Conv2D(64, (3,3), data_format='channels_first'))   
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

# model.add(Conv1D(64, (1)))      ### Hiden Layer2
model.add(Conv2D(64, (3,3), data_format='channels_first'))   
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())

model.add(Dense(512))   ### 全连接层

model.add(Dense(3))     ### 输出层，3维表示结果倾向left、none或者right
model.add(Activation('softmax'))        ###分类问题学习时输出层激活函数常选用softmax函数（回归问题常选用恒等函数

### 编译 ###
model.compile(loss='categorical_crossentropy',      ### 损失函数选用交叉熵，与输出层的softmax函数搭配；如输出层采用恒等函数，则此处应选用平方和误差
              optimizer='adam',     ###优化算法，即如何调整参数
              metrics=['accuracy'])     ###指标？


### 训练 ###
# epochs 训练次数，一个epoch指将样本中的数据全部学习了一次
epochs = int(sys.argv[1])
# batch_size 对样本总数进行分批，以批为单位进行学习，而不是单个数据。由于之前步骤中已经将数据打乱，等效于此处随机分批
batch_size = 32
for epoch in range(epochs):
    model.fit(train_X, train_y, batch_size=batch_size, epochs=1, validation_data=(test_X, test_y))
    score = model.evaluate(test_X, test_y, batch_size=batch_size)
    #print(score)
    MODEL_NAME = f"new_models/{round(score[1]*100,2)}-acc-64x3-batch-norm-{epoch}epoch-{int(time.time())}-loss-{round(score[0],2)}.model"
    model.save(MODEL_NAME)
print("saved:")
print(MODEL_NAME)