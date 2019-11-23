import sys
import keras as K
import tensorflow as tf
from pandas import read_csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D,BatchNormalization
import numpy as np
from keras.layers import *
from keras.models import Model
from keras.models import load_model
from deal_data.cut_label import cut_letter
from deal_data.load_data import load_data
from deal_data.generate_load_data import generate_load_data
from keras.utils import to_categorical
from deal_data.add_data import add_all_class_in_mem_return_ori_and_add_data

model = load_model('/data/wuchenxi/allmodel/new_simeck_model/54080_circle_v1_1_1/model/54080_circle_v1_1_1.hdf5')
# /data/wuchenxi/allmodel/new_simeck_model/15000_1_1_v1/model/15000_1_1_v1.hdf5
# /data/wuchenxi/allmodel/new_simeck_model/54080_1_1_v1/model/54080_1_1_v1.hdf5
model.summary()

print('开始加载数据')

# data_to_train,label_to_train_lb = add_all_class_in_mem_return_ori_and_add_data(signal_data_path='/data/wuchenxi/new_simeck_data/signal54400_circle/signal_320_circle/',
#                                                                                        label_path='/data/wuchenxi/new_simeck_data/signal54400_circle/new_simeck_320.txt',
#                                                                                        which_line=0,
#                                                                                        which_letter=0,
#                                                                                        key_length=4*320,
#                                                                                        each_class_number=10,
#                                                                                        choose_number=2)


data_to_train = load_data('/data/wuchenxi/new_simeck_data/signal54400_circle_v2/signal_320_v3/')
# data_to_train = data_to_train[10000:13000]


label_to_train = cut_letter('/data/wuchenxi/new_simeck_data/signal54400_circle/new_simeck_320.txt', 1,1, 4*320)
label_to_train_lb = to_categorical(label_to_train, 16)



data_to_train = np.expand_dims(data_to_train, axis=2)

eval2 = model.evaluate(data_to_train,label_to_train_lb,
                                 verbose=1,
                                 )

"""
all 1 key_1_1
320/320 [==============================] - 7s 23ms/step
acc: [0.034768438152968886, 1.0]

"""



"""
3000/3000 [==============================] - 38s 13ms/step
acc: [0.17014274275302887, 0.9533333331743876]

"""

"""
320_1
320/320 [==============================] - 7s 22ms/step
acc: [1.0508499547839165, 0.90625]
/data/wuchenxi/allmodel/new_simeck_model/15000_1_1_v1/model/15000_1_1_v1.hdf5
320/320 [==============================] - 7s 22ms/step
acc: [1.4667805671691894, 0.475]


"""

"""
320_2
320/320 [==============================] - 7s 22ms/step
acc: [1.0186179928481578, 0.91875]


"""

"""

"""

# eval2 = model.evaluate_generator(generate_load_data('/data/wuchenxi/new_simeck_data/signal13533/',
#                                           '/data/wuchenxi/new_simeck_data/new_simeck_13533.txt',1,1,4*13533),
#                                  verbose=1,
#                                  steps=10000)
"""
acc: [0.16831567590018093, 0.9547]
"""
print('acc:',eval2)