import time
import serial



class BluetoothSerial:
    def __init__(self):
        self.serial = serial.Serial('COM4', 9600)
        if self.serial.isOpen():
            print("open success")
        else:
            print("open failed")
        self.last_act = "none"

    def recv(self):
        while True:
            data = self.serial.read_all()
            if data == '':
                continue
            else:
                break
        return data

    def send(self, action):
        if action == "keep":
            action = self.last_act
        if action == "left":
            self.serial.write(b'\x03')
        elif action == "right":
            self.serial.write(b'\x04')
        elif action == "up":
            self.serial.write(b'\x01')
        elif action == "down":
            self.serial.write(b'\x02')
        elif action == "none":
            self.serial.write(b'\x00')
        else:
            return 0
        self.last_act = action
        time.sleep(0.5)
        self.serial.write(b'\x00')
    
    def close(self):
        self.serial.close()


if __name__ == "__main__":
    blt = BluetoothSerial()
    blt.send("up")
    data = blt.recv()
    print("receive : ",data)
    time.sleep(0.5)
    # bluetooth_serial.write(b'\x00')
    blt.send("none")
    data = blt.recv()
    print("receive : ",data)
    time.sleep(0.5)
    # bluetooth_serial.write(b'\x02')
    blt.send("down")
    data = blt.recv()
    print("receive : ",data)
    time.sleep(0.5)
    # bluetooth_serial.write(b'\x00')
    blt.send("none")
    data = blt.recv()
    print("receive : ",data)

    blt.close()
