# This file is wrote to make the EEG data (for training or testing)
# When the code below run
# You can control a Box in graphic window moving
# turn left, right or just stay by your keyboard or controller
# while this program will record your brain wave information

# Keyboard & Controller can help you focus your mind keep on the control information
# However, they can also be the confusion covering the valuable feature of EEG

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

conf = ConfigReader("hyperParameters.ini", "9c")  #数据总是以9分类形式录制
reshape = (-1, 8, 60)
FFT_MAX_HZ = conf.frequency_slot     #fft频率区间
HM_SECONDS = 10  # this is approximate. Not 100%. do not depend on this.        #HM是什么意思？
TOTAL_ITERS = HM_SECONDS*25  # ~25 iters/sec        #采样频率？
ACTION = sys.argv[1] # THIS IS THE ACTION YOU'RE THINKING (left, none or right)
# 此处在读取action后，
# 应从hyperParameters中读取ACTIONS列表，
# 验证其中是否已存在本action，
# 如不存在，应将其添加
## 不将action限定在left,right,none是打算通过提高分类数量来提高最终准确率
conf.addActions(ACTION)

last_print = time.time()
fps_counter = deque(maxlen=150)

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')
# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

box = BoxGraphicView()

correct = 0 
channel_datas = []

for i in range(TOTAL_ITERS):  # how many iterations. Eventually this would be a while True
    channel_data = []
    for i in range(8): # each of the 8 channels here
        sample, timestamp = inlet.pull_sample()
        channel_data.append(sample[:FFT_MAX_HZ])    #频率上限

    fps_counter.append(time.time() - last_print)
    last_print = time.time()
    cur_raw_hz = 1/(sum(fps_counter)/len(fps_counter))
    print(cur_raw_hz)

    box.move('random',2)
    # box.move(ACTION.split("_")[0],1)
    channel_datas.append(channel_data)

# plt.plot(channel_datas[0][0])
# plt.show()

datausage = sys.argv[2]
if datausage == "train":
    datadir = conf.training_data_dir
    datadir_3c = conf.getAttr("3c", "training_data_dir")
elif datausage == "test":
    datadir = conf.testing_data_dir
    datadir_3c = conf.getAttr("3c", "testing_data_dir")

if not os.path.exists(datadir):
    os.mkdir(datadir)
if not os.path.exists(datadir_3c):
    os.mkdir(datadir_3c)

actiondir = f"{datadir}/{ACTION}"
ACTION_3c = ACTION.split("_")[0]
actiondir_3c = f"{datadir_3c}/{ACTION_3c}"
if not os.path.exists(actiondir):
    os.mkdir(actiondir)
if not os.path.exists(actiondir_3c):
    os.mkdir(actiondir_3c)

print(len(channel_datas))

print(f"saving {ACTION} data...")
np.save(os.path.join(actiondir, f"{int(time.time())}.npy"), np.array(channel_datas))
np.save(os.path.join(actiondir_3c, f"{int(time.time())}.npy"), np.array(channel_datas))
print("done.")
winsound.Beep(550,500)
for action in conf.actions:
    # 此处的范围应改为从hyperParameters中读取的ACTIONS列表
    print(f"{action}:{len(os.listdir(f'{datadir}/{action}'))}")
    # print(action, sum(os.path.getsize(f'{datadir}/{action}/{f}') for f in os.listdir(f'{datadir}/{action}'))/1_000_000, "MB")
