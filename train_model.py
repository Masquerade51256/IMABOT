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
ACTIONS = dataLoading.ACTIONS
TIME_SLOT = CONF.time_slot
FREQUENCY_SLOT = CONF.frequency_slot 
CHANNELS_NUM = len(CONF.selected_channels)
RESHAPE = (-1, TIME_SLOT, FREQUENCY_SLOT, CHANNELS_NUM) 
HIDDEN_LAYERS = int(CONF.getAttr("default", "hidden_layers"))
# 用于后续规格化，(NTFC)
# 由于后续keras中卷积层默认通道数在最后一个维度上，即channels_last，故此处需要将8放在最后
# cpu版本的tf不支持channels_first，只支持NHWC模式，即channels_last
OUT_SIZE = len(ACTIONS) #输出规格，与分类数有关

# ========================== data create =======================
print("Loading data...")
train_data, test_data, validate_data = dataLoading.load("new_data")
print("Done.")

train_X, train_y = dataLoading.tag_divide(train_data)
test_X, test_y = dataLoading.tag_divide(test_data)
val_X, val_y = dataLoading.tag_divide(validate_data)

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
model.add(Conv2D(8, 2, input_shape=train_X.shape[1:], activation='relu'))   
### 尝试改为2维卷积核，因为输入数据是二维的（不包括批及通道维度）， 
# input_shape=(50,60,8)(实际输入时会增加批维度，即(batch_size,50,60,8))
# 根据卷积前后形状变化公式：HO=(HI+2padding-HK)/stride+1
# output_shape=(batch_size,48,58,64)
# 若想使卷积不缩小数据规模，即HO=HI，求解得padding=(h-1)/2，此处即padding=1
model.add(MaxPooling2D())

### Hidden_Layers
# for i in range(HIDDEN_LAYERS):
#   model.add(Conv2D(64, 3, padding='same', activation='relu'))
#   model.add(MaxPooling2D())

model.add(Conv2D(16, 2, padding='same', activation='relu'))

model.add(Flatten())
model.add(Dense(32, activation='relu'))   

### Output Layer
model.add(Dense(OUT_SIZE, activation='softmax')) 
###分类问题学习时输出层激活函数常选用softmax函数（回归问题常选用恒等函数
model.summary()


### 编译 ###
model.compile(loss='categorical_crossentropy',      
            ### 损失函数选用交叉熵，与输出层的softmax函数搭配；如输出层采用恒等函数，则此处应选用平方和误差
              optimizer='adam',     
              ###优化算法，即如何调整参数
              metrics=['accuracy'])     
              ###指标？

# ========================== model train ==========================

### 训练 ###
epochs = CONF.epochs
batch_size = CONF.batch_size
model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs, validation_data=(val_X,val_y))
score = model.evaluate(test_X, test_y, batch_size=batch_size)

channels =CONF.getAttr(CONF.confMode, "selected_channels").replace(",","")
MODEL_NAME = f"new_models/{round(score[1]*100,2)}acc-64x3x{HIDDEN_LAYERS}-{batch_size}batchsize-{epochs}epochs-{round(score[0],2)}loss-{channels}-{int(time.time())}.model"
model.save(MODEL_NAME)
print("channels: ", CONF.selected_channels)
print("saved:", MODEL_NAME)
