import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization
import time
import dataLoading
import gc
from configReader import ConfigReader

CONF = ConfigReader("hyperParameters.ini")
ACTIONS = CONF.actions
TIME_SLOT = CONF.time_slot
FREQUENCY_SLOT = CONF.frequency_slot
CHANNELS_NUM = CONF.channels_num
RESHAPE = (-1, TIME_SLOT, FREQUENCY_SLOT, CHANNELS_NUM) 
# 用于后续规格化，(NTFC)
# 由于后续keras中卷积层默认通道数在最后一个维度上，即channels_last，故此处需要将8放在最后
# cpu版本的tf不支持channels_first，只支持NHWC模式，即channels_last
OUT_SIZE = len(ACTIONS) #输出规格，与分类数有关

# ========================== data create =======================

print("Loading data...")
all_data = dataLoading.load(CONF.getAttr("3c", "data_dir"))
print("Done.")
data_size = len(all_data)
print("all data: ", data_size)

print("Making test data...")
test_size = int(data_size*0.2)
test_data, train_data = np.split(np.array(all_data,dtype=object),[test_size])
del(all_data)
gc.collect()
test_X, test_y = dataLoading.tag_divide(test_data)
del(test_data)
gc.collect()
print("Done.")
print("test data: ", test_X.shape)

print("Making train data...")
train_X, train_y = dataLoading.tag_divide(train_data)
del(train_data)
gc.collect()
print("Done.")
print("train data: ", train_X.shape)

# ========================== model design ==========================

### 选择模型类型 ###
model = Sequential()

### 构建各层网络 ###
### Input Layer
# model.add(Conv1D(64, (3), input_shape=train_X.shape[1:]))
# 上面一行为模型增加了一个Conv1D作为第一个层，即输入层
# Conv1D的参数中，filter=64，表示卷积核的数量，同时也是输出特征图的通道数  
# kernel_size=(3)，代表卷积核的大小
# 由于为输入层，所以还要提供input_shape参数，[1:]表示取该元组的第二（标号为1）到最后一个元素，1代表批的形状是一维的
model.add(Conv2D(64, (3,3), input_shape=train_X.shape[1:]))   
### 尝试改为2维卷积核，因为输入数据是二维的（不包括批及通道维度）， 
# input_shape=(50,60,8)(实际输入时会增加批维度，即(batch_size,50,60,8))
# 根据卷积前后形状变化公式：HO=(HI+2padding-HK)/stride+1
# output_shape=(batch_size,48,58,64)
# 若想使卷积不缩小数据规模，即HO=HI，求解得padding=(h-1)/2，此处即padding=1
model.add(Activation('relu'))       ### 激发函数

### Hidden_Layer1
model.add(Conv2D(64, (3,3)))   
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

### Hidden Layer2
model.add(Conv2D(64, (3,3)))   
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

### Hidden Layer3
model.add(Conv2D(64, (3,3)))   
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(512))   

### Output Layer
model.add(Dense(OUT_SIZE))
model.add(Activation('softmax'))        
###分类问题学习时输出层激活函数常选用softmax函数（回归问题常选用恒等函数

### 编译 ###
model.compile(loss='categorical_crossentropy',      
            ### 损失函数选用交叉熵，与输出层的softmax函数搭配；如输出层采用恒等函数，则此处应选用平方和误差
              optimizer='adam',     
              ###优化算法，即如何调整参数
              metrics=['accuracy'])     
              ###指标？

# ========================== model train ==========================

### 训练 ###
# epochs 训练次数，一个epoch指将样本中的数据全部学习了一次
epochs = CONF.epochs
# batch_size 对样本总数进行分批，以批为单位进行学习，而不是单个数据。由于之前步骤中已经将数据打乱，等效于此处随机分批
batch_size = CONF.batch_size
model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs, validation_split=0.25)
score = model.evaluate(test_X, test_y, batch_size=batch_size)
MODEL_NAME = f"{CONF.models_dir}/{round(score[1]*100,2)}-acc-64x3-batch-norm-{int(time.time())}-loss-{round(score[0],2)}.model"
model.save(MODEL_NAME)
print("saved:")
print(MODEL_NAME)
