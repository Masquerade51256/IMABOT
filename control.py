from mne.decoding import CSP
from pylsl import StreamInlet, resolve_stream
import numpy as np
import os
import load_rawdata
import tensorflow as tf
from blt import BluetoothSerial

CONF = load_rawdata.CONF
TIME_SLOT = CONF.time_slot
DATA_FOR_DECISSION = 10
CONTROL_SLOT = TIME_SLOT+DATA_FOR_DECISSION-1
ACTIONS = load_rawdata.ACTIONS
MODEL_NAME = "38.89acc-1.34loss-1621049841.model"
MODEL_PATH = os.path.join("csp_models", MODEL_NAME)
MODEL = tf.keras.models.load_model(MODEL_PATH)
# 打开蓝牙串口
BLT = BluetoothSerial()


def decide(outputs):
    acc = float(MODEL_NAME[0:4])/100
    print(acc)
    actions = np.array(ACTIONS)
    t = np.array([0,0,0,0,0])
    for item in outputs:
        print(item)
        t[np.argmax(item)] += 1
    print(t)  
    total = len(outputs)
    t= t/total
    print(t)
    t= t-acc
    print(t)
    t= t*t
    print(t)
    print(np.min(t))
    print(np.argmin(t))
    eq = len(np.argwhere((t - np.min(t))<=1e-10))
    print(eq)
    if eq < len(actions)/2:
        act = actions[np.argmin(t)]
    else:
        act = "keep"
    return act


if __name__ =="__main__":

    # 接收推流数据
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    # 创建inlet读取数据
    inlet = StreamInlet(streams[0])

    csp = CSP(n_components=4, reg=None, log=False, norm_trace=False)
    csp_name = "1621049839"
    csp.filters_ = np.load(os.path.join("csp_estimator",csp_name,"filters.npy"))
    csp.patterns_ = np.load(os.path.join("csp_estimator",csp_name,"patterns.npy"))
    csp.mean_ = np.load(os.path.join("csp_estimator",csp_name,"mean.npy"))
    csp.std_ = np.load(os.path.join("csp_estimator",csp_name,"std.npy"))

    datas = []
    while(1):
        sample, _ = inlet.pull_sample()
        datas.append(sample)
        l = len(datas)
        if l >= CONTROL_SLOT:
            head = l-CONTROL_SLOT
            raw_data = np.array(datas).reshape(-1,8)[head:]
            cut_data = load_rawdata.cutData(raw_data)
            input_data = csp.transform(cut_data)
            _, c = input_data.shape
            input_data = input_data.reshape(-1,c,1)
            outputs = MODEL.predict(input_data)
            action = decide(outputs)
            BLT.send(action)
        else:
            BLT.send("none")

