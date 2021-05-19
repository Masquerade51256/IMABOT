import numpy as np
from configReader import ConfigReader
import tensorflow as tf
import os
import dataLoading
from boxGraphicView import BoxGraphicView
from time import sleep

def test(data,tag,actions,model):
    total = len(data)
    print(total)
    correct = 0
    output = model.predict(data)
    box = BoxGraphicView()
    for i in range(total):
        act = actions[np.argmax(output[i])]
        print(act, output[i])
        box.move(act,step=1)
        sleep(0.03)
        if act == tag:
            correct += 1
    acc = correct/total
    return acc

if __name__ == "__main__":
    CONF = ConfigReader()
    actions = CONF.actions

    model_name = CONF.test_model
    model_dir = os.path.join(CONF.models_dir,model_name)
    model = tf.keras.models.load_model(model_dir)

    tag = "left"
    act = os.path.join("horizontal_data",tag)
    file_name = "1618882899.npy"
    data_dir = os.path.join(act,file_name)
    data = np.load(data_dir)
    data = dataLoading.cutData(data,stride=1)

    print("accuracy:", test(data,tag,actions,model))
    print("data:", tag, file_name)
    print("model:", model_name)
