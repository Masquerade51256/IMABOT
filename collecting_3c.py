from pylsl import StreamInlet, resolve_stream
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import style
from collections import deque
import os
import sys
import winsound
from BoxGraphicView import BoxGraphicView
from configReader import ConfigReader

conf = ConfigReader("hyperParameters.ini", "3c")
reshape = (-1, 8, 60)
FFT_MAX_HZ = conf.frequency_slot
HM_SECONDS = 10
TOTAL_ITERS = HM_SECONDS*25
ACTION = sys.argv[1]
CHANNELS_NUM = conf.channels_num
conf.addActions(ACTION)

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
    # print(l)
    if l >=TIME_SLOT:
        head = l-TIME_SLOT
        raw_data = np.array(channel_datas).reshape(RESHAPE)[head:]
        # print(raw_data)
        input_data = dataLoading.format(raw_data)
        # print(input_data)
        output_data = model.predict(input_data)
        # print(output_data)
        output_act = ACTIONS[np.argmax(output_data)]
        # print(output_act)
        act = output_act.split("_")[0]
        box.move(act,1)
        if act == ACTION:
            correct += 1
        total += 1
    else:
        box.move("random",1)

# plt.plot(channel_datas[0][0])
# plt.show()

datausage = sys.argv[2]
if datausage == "train":
    datadir_3c = conf.training_data_dir
    datadir_date = time.strftime("%Y%m%d", time.localtime())+"_train_data"
elif datausage == "test":
    datadir_3c = conf.testing_data_dir

if not os.path.exists(datadir_3c):
    os.mkdir(datadir_3c)

ACTION_3c = ACTION.split("_")[0]
actiondir_3c = f"{datadir_3c}/{ACTION_3c}"
if not os.path.exists(actiondir_3c):
    os.mkdir(actiondir_3c)

print(len(channel_datas))

print(f"saving {ACTION} data...")
np.save(os.path.join(actiondir_3c, f"{int(time.time())}.npy"), np.array(channel_datas))
print("done.")
# winsound.Beep(550,500)
for action in conf.actions:
    print(f"{action}:{len(os.listdir(f'{datadir_3c}/{action}'))}")
print(ACTION, correct/total)