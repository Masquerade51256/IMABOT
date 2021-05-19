import os
from configReader import ConfigReader

CONF = ConfigReader()

for j in range(2):
    if j == 0:
        mode = "vertical"
    elif j == 1:
        mode = "horizontal"
    CONF.setMode(mode)
    for i in range(5):
        if i == 0:
            c_str = "0,1,2,3,4,5,6,7"
        elif i == 1:
            c_str = "2,3,4,5,6,7"
        elif i == 2:
            c_str = "0,1,4,5,6,7"
        elif i == 3:
            c_str = "0,1,2,3,6,7"
        elif i == 4:
            c_str = "0,1,2,3,4,5"
        else:
            c_str = ""
    # c_str = "0,1,2,3,4,5,6,7"
        CONF.setChannels(c_str)
        os.system("python train_model.py")
