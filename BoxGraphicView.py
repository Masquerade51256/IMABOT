import random
from time import sleep
import cv2
import numpy as np

class BoxGraphicView:
    def __init__(self):
        '''
        e.g.:\n
        box=BoxGraphicView()\n
        for i in range(100):\n
            sleep(0.1)\n
            box.move('random',3)\n
        '''
        self.WIDTH = 800
        self.HEIGHT = 800
        self.SQ_SIZE = 50
        self.MOVE_SPEED = 1
        self.square = {'x1': int(int(self.WIDTH)/2-int(self.SQ_SIZE/2)), 
          'x2': int(int(self.WIDTH)/2+int(self.SQ_SIZE/2)),
          'y1': int(int(self.HEIGHT)/2-int(self.SQ_SIZE/2)),
          'y2': int(int(self.HEIGHT)/2+int(self.SQ_SIZE/2))}
        self.box = np.ones((self.square['y2']-self.square['y1'], self.square['x2']-self.square['x1'], 3)) * np.random.uniform(size=(3,))
        self.horizontal_line = np.ones((self.HEIGHT, 10, 3)) * np.random.uniform(size=(3,))
        self.vertical_line = np.ones((10, self.WIDTH, 3)) * np.random.uniform(size=(3,))
        self.env = np.zeros((self.WIDTH, self.HEIGHT, 3))

    def move(self,dir_h="none",dir_v="none",step=-1):
        '''
        e.g.:\n
        box=BoxGraphicView()\n
        for i in range(100):\n
            sleep(0.1)\n
            box.move('random',3)\n
        '''
        self.env = np.zeros((self.WIDTH, self.HEIGHT, 3))
        self.env[:,self.HEIGHT//2-5:self.HEIGHT//2+5,:] = self.horizontal_line
        self.env[self.WIDTH//2-5:self.WIDTH//2+5,:,:] = self.vertical_line
        self.env[self.square['y1']:self.square['y2'], self.square['x1']:self.square['x2']] = self.box
        cv2.imshow('', self.env)
        cv2.waitKey(1)
        if step == -1:
            step = self.MOVE_SPEED

        if dir_h == 'left':
            self.square['x1']-=step
            self.square['x2']-=step
        elif dir_h == 'right':
            self.square['x1']+=step
            self.square['x2']+=step
        elif dir_h == 'none':
            pass
        else:
            move = random.choice([-1,0,1])
            self.square['x1']+=(move*step)
            self.square['x2']+=(move*step)

        if dir_v == "up":
            self.square['y1']+=step
            self.square['y2']+=step
        elif dir_v == "down":
            self.square['y1']-=step
            self.square['y2']+=step
        elif dir_v == 'none':
            pass
        else:
            move = random.choice([-1,0,1])
            self.square['y1']+=(move*step)
            self.square['y2']+=(move*step)

if __name__=="__main__":
    box=BoxGraphicView()
    for i in range(100):
        sleep(0.1)
        box.move('random',3)