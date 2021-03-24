import numpy as np
import tensorflow as tf
import os
import dataLoading
from configReader import ConfigReader

CONF = ConfigReader("hyperParameters.ini")
CONF.setMode("9c")
ACTIONS = CONF.actions
TIME_SLOT = CONF.time_slot
FREQUENCY_SLOT = CONF.frequency_slot
CHANNELS_NUM = CONF.channels_num
RESHAPE = (-1, TIME_SLOT, FREQUENCY_SLOT, CHANNELS_NUM)  

def map3Accuracy(model:tf.keras.Model, test_X, test_y):
    """
    test_y: 需要map3_str
    """
    total = len(test_X)
    correct = 0
    for i in range(total):
        # print(test_X[i].shape)
        out = model.predict(test_X[i].reshape(RESHAPE))
        print(out)
        out_act = ACTIONS[np.argmax(out)].split("_")[0]
        if out_act == test_y[i]:
            correct += 1
        else: 
            print(test_y[i],"->",out_act)
    return correct/total

if __name__ == "__main__":
    # print(CONF.confMode)
    # print(CONF.testing_data_dir)
    test_X, test_y = dataLoading.load_and_format(CONF.testing_data_dir, tag_flag="map3_str", cfm="9c")
    
    print(len(test_X))

    test_X = np.array(test_X).reshape(RESHAPE)
    test_y = np.array(test_y)

    MODEL_NAME = os.path.join(CONF.models_dir,"37.78-acc-64x3-batch-norm-9epoch-1616551090-loss-3.48.model")
    model = tf.keras.models.load_model(MODEL_NAME)
    print(map3Accuracy(model, test_X, test_y))
