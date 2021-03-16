from pynput.keyboard import Key, Listener
from BoxGraphicView import BoxGraphicView as bv
Value = 0
box = bv()
def on_press(key):
    print('{0} pressed'.format(
        key))

def on_release(key):
    print('{0} release'.format(
        key))
    if key == Key.esc:
        # Stop listener
        return False

def update(key):
    global Value
    if key == Key.right:
        Value += 10
        box.move('right')
    elif key == Key.left:
        Value -= 10
        box.move('left')

    print(Value)

# Collect events until released
with Listener(
        on_press=update,
        on_release=on_release) as listener:
    listener.join()

try:
    Listener.stop
except Exception:
    pass
