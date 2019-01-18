# coding=utf-8

import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras import backend as K
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
import h5py
import cv2
from PIL import Image
import numpy as np
from os.path import join
import os
import os.path

batch_size = 64
epochs = 30
input_shape = (40, 40, 1)
eng_list = list('abcdefghijklmnopqrstuvwxyz0123456789')
code_classes = 36
test_ratio = 0.3

x_list = []
y_list = []

# 读取图片
image_path = ''
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
_, imgt = cv2.threshold(img, 140, 255, cv2.THRESH_BINARY_INV)

x_data = imgt / 255
y_data = ''

x_list.append(x_data)
y_list.append(y_data)

x_train, x_test, y_train, y_test = train_test_split(np.asarry(x_list), np.asarray(y_list),
                                                    test_size=test_ratio,
                                                    random_state=42)

# 网络搭建
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(code_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

print(model.summary())
# 载入网络权重
model.load_weights('xxxx.hdf5')

# 回调函数，每个epoch结束后如果loss是历史最低的，则保存网络参数
checkpointer = ModelCheckpoint(filepath='xxxx.hdf5',
                               verbose=1,
                               save_best_only=True)
# 开始训练
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_test, y_test), callbacks=[checkpointer])

# 网络评价
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

# 保存网络的结构到json文件
json_string = model.to_json()
open('xxxx.json', 'w').write(json_string)
