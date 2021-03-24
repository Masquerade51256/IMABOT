from pylsl import StreamInlet, resolve_stream
import tensorflow as tf
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
import dataLoading

CONF = ConfigReader("hyperParameters.ini", "9c")
RESHAPE = (-1, 8, 60)
FFT_MAX_HZ = CONF.frequency_slot
HM_SECONDS = 10
TOTAL_ITERS = HM_SECONDS*25
ACTIONS = CONF.actions
CHANNELS_NUM = CONF.channels_num
TIME_SLOT = CONF.time_slot

MODEL_NAME = os.path.join(CONF.models_dir,"37.78-acc-64x3-batch-norm-9epoch-1616551090-loss-3.48.model")
model = tf.keras.models.load_model(MODEL_NAME)
model.predict(np.zeros((32,60,60,8)))


last_print = time.time()
fps_counter = deque(maxlen=150)

print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])

box = BoxGraphicView()

channel_datas = []

for i in range(TOTAL_ITERS):  # how many iterations. Eventually this would be a while True
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
        print(raw_data)
        input_data = dataLoading.format(raw_data)   ####此句有错，返回全1数组
        print(input_data)
        output_data = model.predict(input_data)
        # print(output_data)
        output_act = ACTIONS[np.argmax(output_data)]
        # print(output_act)
        act = output_act.split("_")[0]
        box.move(act,1)