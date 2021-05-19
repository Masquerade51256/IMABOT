"""
python collect_rawdata.py left
"""

from pylsl import StreamInlet, resolve_stream
from boxGraphicView import BoxGraphicView
from configReader import ConfigReader
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys


ACTION = sys.argv[1]
mode = "default"
if ACTION == "left" or ACTION == "right":
    mode = "horizontal"
else:
    mode = "vertical"

CONF = ConfigReader("hyperParameters.ini", mode)
RESHAPE = (-1, 8, 60)
FFT_MAX_HZ = CONF.frequency_slot
HM_SECONDS = 10
TOTAL_ITERS = HM_SECONDS*25
CHANNELS_NUM = CONF.channels_num
ACTIONS = ["left", "right", "up", "down", "none"]


# last_print = time.time()
# fps_counter = deque(maxlen=150)

# 接收推流数据
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')
# 创建inlet读取数据
inlet = StreamInlet(streams[0])
#创建视图
box = BoxGraphicView()

channel_datas = []

for i in range(TOTAL_ITERS):
    channel_data = []
    sample, _ = inlet.pull_sample()
    box.move('random',1)

    channel_datas.append(sample)

sample = np.array(channel_datas)
print(sample.shape)


datadir = "raw_data"
datadir_date = "all_data/raw_data/"+time.strftime("%Y%m%d", time.localtime())

if not os.path.exists(datadir):
    os.mkdir(datadir)
if not os.path.exists(datadir_date):
    os.mkdir(datadir_date)


actiondir = f"{datadir}/{ACTION}"
actiondir_date = f"{datadir_date}/{ACTION}"
if not os.path.exists(actiondir):
    os.mkdir(actiondir)
if not os.path.exists(actiondir_date):
    os.mkdir(actiondir_date)

print(len(channel_datas))

print(f"saving {ACTION} data...")
np.save(os.path.join(actiondir, f"{int(time.time())}.npy"), np.array(channel_datas))
np.save(os.path.join(actiondir_date, f"{int(time.time())}.npy"), np.array(channel_datas))
print("done.")
for action in ACTIONS:
    print(f"{action}:{len(os.listdir(f'{datadir}/{action}'))}")