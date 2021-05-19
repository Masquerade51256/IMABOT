from mne.decoding import CSP
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
import load_rawdata
import time
import os

CONF = load_rawdata.CONF
ACTIONS = load_rawdata.ACTIONS
MYCSP = CSP(n_components=8)

def csp_trans(X,y,fit):
    print("===============================debug===============================")
    print("csp_train.py, fit_transform, X")
    print("shape", X.shape)
    # print("data", X)
    print("===============================debug===============================")
    if fit:
        csp_X = MYCSP.fit_transform(X=X,y=y)
    else:
        csp_X = MYCSP.transform(X=X)
    print("csp_X shape", csp_X.shape)
    # print("csp_X", csp_X)
    return csp_X

def onehot_trans(y):
    onehot_y = []
    l = len(ACTIONS)
    for i in y:
        onehot = np.zeros((l,))
        onehot[i] = 1
        onehot_y.append(onehot)
    onehot_y = np.array(onehot_y)
    # print(onehot_y)
    return onehot_y


print("Loading data...")
train_data, test_data, validate_data = load_rawdata.load()
print("Done.")
train_X, train_y = load_rawdata.tag_divide(train_data)
test_X, test_y = load_rawdata.tag_divide(test_data)
val_X, val_y = load_rawdata.tag_divide(validate_data)

train_X = csp_trans(train_X,train_y,fit=True)
test_X = csp_trans(test_X,test_y,fit=None)
val_X = csp_trans(val_X,val_y,fit=None)

csp_path = os.path.join("csp_estimator", str(int(time.time())))
if not os.path.exists(csp_path):
    os.mkdir(csp_path)
filters_ = MYCSP.filters_
np.save(os.path.join(csp_path, "filters"), np.array(filters_))
patterns_ = MYCSP.patterns_
np.save(os.path.join(csp_path, "patterns"), np.array(patterns_))
mean_ = MYCSP.mean_
np.save(os.path.join(csp_path, "mean"), np.array(mean_))
std_ = MYCSP.std_
np.save(os.path.join(csp_path, "std"), np.array(std_))

train_y = onehot_trans(train_y)
test_y = onehot_trans(test_y)
val_y = onehot_trans(val_y)
_, c = train_X.shape

train_X = train_X.reshape(-1,c,1)
test_X = test_X.reshape(-1,c,1)
val_X = val_X.reshape(-1,c,1)

print("train_X shape", train_X.shape)
print(MYCSP)

model = keras.Sequential()
model.add(Conv1D(8, 3, input_shape = train_X.shape[1:], activation='relu'))
model.add(MaxPooling1D())
model.add(Conv1D(8, 3, padding='same', activation='relu'))
model.add(Conv1D(8, 3, padding='same', activation='relu'))
# model.add(MaxPooling1D())

model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_X, train_y, batch_size=32, epochs=10, validation_data=(val_X, val_y))
score = model.evaluate(test_X, test_y, batch_size=32)

MODEL_NAME = f"csp_models/{round(score[1]*100,2)}acc-{round(score[0],2)}loss-{int(time.time())}.model"
model.save(MODEL_NAME)