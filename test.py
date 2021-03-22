import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, Conv2D
from tensorflow.keras.layers import MaxPooling1D, MaxPooling2D, BatchNormalization
import os
import random
import sys
import time
import dataLoading
from configReader import ConfigReader

CONF = ConfigReader("hyperParameters.ini")
ACTIONS = CONF.actions
TIME_SLOT = CONF.time_slot
FREQUENCY_SLOT = CONF.frequency_slot
CHANNELS_NUM = CONF.channels_num
reshape = (-1, TIME_SLOT, FREQUENCY_SLOT, CHANNELS_NUM)  

def map3Accuracy(model:tf.keras.Model, test_X, test_y):
    """
    test_y: 需要map3_str
    """
    total = len(test_X)
    correct = 0
    for i in range(total):
        out = model.predict(test_X[i].reshape(reshape))
        # print(out)
        out_act = ACTIONS[np.argmax(out)].split("_")[0]
        if out_act == test_y[i]:
            correct += 1
        else: 
            print(test_y[i],"->",out_act)
    return correct/total
    pass

if __name__ == "__main__":
    testdata = dataLoading.load_and_format(CONF.testing_data_dir, tag_flag="map3_str")
    test_X = []
    test_y = []
    for X, y in testdata:
        test_X.append(X)
        test_y.append(y)

    print(len(test_X))

    test_X = np.array(test_X).reshape(reshape)
    test_y = np.array(test_y)

    MODEL_NAME = os.path.join('new_models',"34.17-acc-64x3-batch-norm-9epoch-1616395610-loss-2.41.model")  # your model path here. 
    model = tf.keras.models.load_model(MODEL_NAME)
    print(map3Accuracy(model, test_X, test_y))