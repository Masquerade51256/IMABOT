import configparser
import os


class ConfigReader:
    def __init__(self, confPath="hyperParameters.ini",cfm = "default"):
        self.conf = configparser.ConfigParser()
        self.conf.read(confPath)
        if cfm == "default":
            self.confMode = self.conf["default"]["mode"]
        else:
            self.confMode = cfm
        self.training_data_dir = self.conf[self.confMode]["training_data_dir"]
        self.testing_data_dir = self.conf[self.confMode]["testing_data_dir"]
        self.models_dir = self.conf[self.confMode]["models_dir"]
        self.actions = self.conf[self.confMode]["actions"].split(',')
        self.channels_num = int(self.conf["default"]["channels_num"])
        self.frequency_slot = int(self.conf["default"]["frequency_slot"])
        self.time_slot = int(self.conf["default"]["time_slot"])
        self.data_stride = int(self.conf["default"]["data_stride"])
        self.batch_size = int(self.conf["default"]["batch_size"])
        self.epochs = int(self.conf["default"]["epochs"])

    def addActions(self, action, actionMode="default"):
        if actionMode == "default":
            actionMode = self.confMode
        if action not in self.actions:
            self.actions.append(action)
            a_str = ""
            for act in self.actions:
                a_str+=act
                a_str+=','
            a_str=a_str.strip(',')
            self.conf.set(actionMode, "actions", a_str)
            
            with open('hyperParameters.ini', 'w') as configfile:
                self.conf.write(configfile)
            return 1
        else:
            return 0
    
    def getAttr(self, section, item):
        return self.conf[section][item]

    # def setMode(self, mode):
    #     if self.confMode != mode:
    #         self.confMode = mode
    #         self.conf.set("default", "mode", self.confMode)
    #         with open('hyperParameters.ini', 'w') as configfile:
    #             self.conf.write(configfile)
    #         return 1
    #     else:
    #         return 0


if __name__ == "__main__":
    confR = ConfigReader()
    actions = confR.actions
    print(actions)
    confR.addActions("up") 
    print(confR.batch_size)
    print(type(confR.batch_size))