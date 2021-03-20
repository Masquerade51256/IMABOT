import configparser
import os


class ConfigReader:
    def __init__(self, confPath="hyperParameters.ini"):
        self.conf = configparser.ConfigParser()
        self.conf.read(confPath)
        self.training_data_dir = self.conf["default"]["training_data_dir"]
        self.testing_data_dir = self.conf["default"]["testing_data_dir"]
        self.models_dir = self.conf["default"]["models_dir"]
        self.actions = self.conf["default"]["actions"].split(',')
        self.channels_num = int(self.conf["default"]["channels_num"])
        self.frequency_slot = int(self.conf["default"]["frequency_slot"])
        self.time_slot = int(self.conf["default"]["time_slot"])
        self.data_stride = int(self.conf["default"]["data_stride"])
        self.batch_size = int(self.conf["default"]["batch_size"])
        self.epochs = int(self.conf["default"]["epochs"])

    def addActions(self, action):
        if action not in self.actions:
            self.actions.append(action)
            a_str = ""
            for act in self.actions:
                a_str+=act
                a_str+=','
            print(a_str)
            a_str=a_str.rstrip(',')
            print(a_str)
            self.conf.set("default", "actions", a_str)
            
            with open('hyperParameters.ini', 'w') as configfile:
                self.conf.write(configfile)
            return 1
        else:
            return 0



if __name__ == "__main__":
    confR = ConfigReader()
    actions = confR.actions
    print(actions)
    confR.addActions("up")
    print(confR.batch_size)
    print(type(confR.batch_size))