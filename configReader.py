import configparser
import os

conf = configparser.ConfigParser()
conf.read("hyperParameters.ini", encoding="utf-8")

actions = conf["default"]["ACTIONS"]
print(actions)
print(actions[0])
print(type(actions))