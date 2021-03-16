import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
import os
import random
import time
import sys


ACTIONS = ["left", "right", "none"]
reshape = (-1, 8, 60)

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
        training_data[action] = training_data[action][:min(lengths)]        ## 规格化，保证训练样本中的比例为1:1:1 ###

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
model.add(Conv1D(64, (3), input_shape=train_X.shape[1:]))       ### 输入层 input_shape = [8, 60]
model.add(Activation('relu'))       ### 激发函数

model.add(Conv1D(64, (2)))      ### Hiden_Layer1
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(2)))

model.add(Conv1D(64, (1)))      ### Hiden Layer2
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=(2)))

model.add(Flatten())

model.add(Dense(512))

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