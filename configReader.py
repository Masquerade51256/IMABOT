import configparser

class ConfigReader:
    def __init__(self, confPath="hyperParameters.ini",cfm = "default"):
        self.conf = configparser.ConfigParser()
        self.conf.read(confPath)
        if cfm == "default": 
            self.confMode = self.conf["default"]["mode"]
        else:
            self.confMode = cfm
        self.data_dir = self.conf[self.confMode]["data_dir"]
        self.models_dir = self.conf[self.confMode]["models_dir"]
        self.actions = self.conf[self.confMode]["actions"].split(',')
        self.test_model = self.conf[self.confMode]["test_model"]
        self.channels_num = int(self.conf["default"]["total_channels"])
        self.frequency_slot = int(self.conf["default"]["frequency_slot"])
        self.time_slot = int(self.conf["default"]["time_slot"])
        self.data_stride = int(self.conf["default"]["data_stride"])
        self.batch_size = int(self.conf["default"]["batch_size"])
        self.epochs = int(self.conf["default"]["epochs"])

        s = self.conf[self.confMode]["selected_channels"]
        self.selected_channels = s.split(",")
        for i in range(len(self.selected_channels)):
            self.selected_channels[i] = int(self.selected_channels[i])

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
    
    def setMode(self, cfm):
        if self.confMode == cfm:
            return 0
        else:
            self.confMode = cfm
            self.conf.set("default", "mode", self.confMode)
            with open('hyperParameters.ini', 'w') as configfile:
                    self.conf.write(configfile)
            self.__init__()
            return 1

if __name__ == "__main__":
    confR = ConfigReader()
    actions = confR.actions
    print(actions)
    confR.addActions("up") 
    print(confR.batch_size)
    print(type(confR.batch_size))