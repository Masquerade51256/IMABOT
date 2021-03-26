from collections import deque
from pylsl import StreamInlet, resolve_stream
from boxGraphicView import BoxGraphicView
from configReader import ConfigReader
import numpy as np
import tensorflow as tf
import time
import os
import sys
import dataLoading


CONF = ConfigReader("hyperParameters.ini", "3c")
RESHAPE = (-1, 8, 60)
FFT_MAX_HZ = CONF.frequency_slot
HM_SECONDS = 10
TOTAL_ITERS = HM_SECONDS*25
ACTION = sys.argv[1]
CHANNELS_NUM = CONF.channels_num
TIME_SLOT = CONF.time_slot
ACTIONS = CONF.actions
CONF.addActions(ACTION)
MODEL_NAME = os.path.join(CONF.models_dir,"56.6-acc-64x3-batch-norm-9epoch-1616553967-loss-1.77.model")
model = tf.keras.models.load_model(MODEL_NAME)
model.predict(np.zeros((32,60,60,8)))


last_print = time.time()
fps_counter = deque(maxlen=150)

# 接收推流数据
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')
# 创建inlet读取数据
inlet = StreamInlet(streams[0])
#创建视图
box = BoxGraphicView()

total = 0
correct = 0 
channel_datas = []

for i in range(TOTAL_ITERS):
    channel_data = []
    for i in range(CHANNELS_NUM): # 8 channels
        sample, timestamp = inlet.pull_sample()
        channel_data.append(sample[:FFT_MAX_HZ])    #频率上限

    # fps_counter.append(time.time() - last_print)
    # last_print = time.time()
    # cur_raw_hz = 1/(sum(fps_counter)/len(fps_counter))
    # print(cur_raw_hz)

    channel_datas.append(channel_data)
    l = len(channel_datas) 
    print(l)
    if l >=TIME_SLOT:
        head = l-TIME_SLOT
        raw_data = np.array(channel_datas).reshape(RESHAPE)[head:]
        input_data = dataLoading.format(raw_data)
        output_data = model.predict(input_data)
        output_act = ACTIONS[np.argmax(output_data)]
        act = output_act.split("_")[0]
        print(act,output_data)
        box.move(act,1)
        if act == ACTION:
            correct += 1
        total += 1
    else:
        box.move(ACTION,1)

# plt.plot(channel_datas[0][0])
# plt.show()

datausage = sys.argv[2]
if datausage == "train":
    datadir_3c = CONF.training_data_dir
    datadir_date = "all_data/"+time.strftime("%Y%m%d", time.localtime())+"_train_data"
elif datausage == "test":
    datadir_3c = CONF.testing_data_dir
    datadir_date = "all_data/"+time.strftime("%Y%m%d", time.localtime())+"_test_data"

if not os.path.exists(datadir_3c):
    os.mkdir(datadir_3c)
if not os.path.exists(datadir_date):
    os.mkdir(datadir_date)


actiondir_3c = f"{datadir_3c}/{ACTION}"
actiondir_date = f"{datadir_date}/{ACTION}"
if not os.path.exists(actiondir_3c):
    os.mkdir(actiondir_3c)
if not os.path.exists(actiondir_date):
    os.mkdir(actiondir_date)

print(len(channel_datas))

print(f"saving {ACTION} data...")
np.save(os.path.join(actiondir_3c, f"{int(time.time())}.npy"), np.array(channel_datas))
np.save(os.path.join(actiondir_date, f"{int(time.time())}.npy"), np.array(channel_datas))
print("done.")
# winsound.Beep(550,500)
for action in CONF.actions:
    print(f"{action}:{len(os.listdir(f'{datadir_3c}/{action}'))}")
print(ACTION, correct/total)