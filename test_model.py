from pylsl import StreamInlet, resolve_stream
import tensorflow as tf
import numpy as np
import time
from collections import deque
import os
from boxGraphicView import BoxGraphicView
from configReader import ConfigReader
import dataLoading

CONF = ConfigReader("hyperParameters.ini")
RESHAPE = (-1, 8, 60)
FFT_MAX_HZ = CONF.frequency_slot
HM_SECONDS = 10
TOTAL_ITERS = HM_SECONDS*25
ACTIONS = CONF.actions
CHANNELS_NUM = CONF.channels_num
TIME_SLOT = CONF.time_slot

MODEL_NAME = os.path.join(CONF.models_dir,CONF.getAttr("default","test_model"))
print(MODEL_NAME)
model = tf.keras.models.load_model(MODEL_NAME)
model.predict(np.zeros((32,60,60,8)))


last_print = time.time()
fps_counter = deque(maxlen=150)

print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])

box = BoxGraphicView()

channel_datas = []

# for i in range(TOTAL_ITERS):  # how many iterations. Eventually this would be a while True
while(True):
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
        input_data = dataLoading.cutData(raw_data)
        input_data = input_data[...,CONF.selected_channels]
        # print(input_data)


        output_data = model.predict(input_data)
        output_act = ACTIONS[np.argmax(output_data)]
        print(output_act, output_data)
        act = output_act.split("_")[0]
        box.move(act,1)
    else:
        box.move("random",1)