import sys
import keras as K
from keras import backend as Kb
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
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,History,TensorBoard,CSVLogger
from keras.metrics import top_k_categorical_accuracy
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
from keras.callbacks import Callback
import os
from sklearn.metrics import confusion_matrix,f1_score,roc_curve,auc
import matplotlib.pyplot as plt
from keras.utils import multi_gpu_model
from deal_data.cut_label import cut_letter
from deal_data.load_data import load_data


# def BLOCK(seq, filters): # 定义网络的Block
#     cnn = Conv1D(filters*2, 3, padding='SAME', dilation_rate=1, activation='relu')(seq)
#     cnn = BatchNormalization(axis=1)(cnn)
#     cnn = Lambda(lambda x: x[:,:,:filters] + x[:,:,filters:])(cnn)
#     cnn = Conv1D(filters*2, 3, padding='SAME', dilation_rate=2, activation='relu')(cnn)
#     cnn = BatchNormalization(axis=1)(cnn)
#     cnn = Lambda(lambda x: x[:,:,:filters] + x[:,:,filters:])(cnn)
#     cnn = Conv1D(filters*2, 3, padding='SAME', dilation_rate=4, activation='relu')(cnn)
#     cnn = BatchNormalization(axis=1)(cnn)
#     cnn = Lambda(lambda x: x[:,:,:filters] + x[:,:,filters:])(cnn)
#     if int(seq.shape[-1]) != filters:
#         seq = Conv1D(filters, 1, padding='SAME')(seq)
#     seq = add([seq, cnn])
#     return seq

class EvaluateInputTensor(Callback):


    def __init__(self, model,test_X,test_Y,test_log, metrics_prefix='val', verbose=1):
        # parameter of callbacks passed during initialization
        # pass evalation mode directly
        # 初始化传递模式中的回调参数
        super(EvaluateInputTensor, self).__init__()
        self.val_model = model
        self.testx = test_X
        self.testy = test_Y
        self.file_log   = open(test_log,'w')
        self.verbose = verbose
        self.metrics_prefix = metrics_prefix

    def on_epoch_end(self, epoch, logs={}):
        self.val_model.set_weights(self.model.get_weights())
        results = self.val_model.evaluate(self.testx, self.testy,
                                          verbose=self.verbose)
        metrics_str = '\n'
        for result, name in zip(results, self.val_model.metrics_names):
            metric_name = self.metrics_prefix + '_' + name
            logs[metric_name] = result
            if self.verbose > 0:
                metrics_str = metrics_str + metric_name + ': ' + str(result) + ' '

        if self.verbose > 0:
            print(metrics_str)
            self.file_log.writelines(metrics_str+'\n')

def vgg16_model(trainFilePath,testFilePath,batch_size,epochs,name,lr,key_file,which_line,which_letter,key_length,test_size,load_weight=False,weight_path = None,evalONtest = True):
    input_tensor = Input(shape=(35000, 1))
    seq = input_tensor
    seq = SeparableConv1D(64,3,padding='same',activation=None)(seq)
    seq = BatchNormalization(axis=1)(seq)
    seq = ReLU()(seq)
    seq = SeparableConv1D(64,3,padding='same',activation=None)(seq)
    seq = BatchNormalization(axis=1)(seq)
    seq = ReLU()(seq)
    seq = MaxPooling1D(pool_size=2,strides=2)(seq)

    seq = SeparableConv1D(128,3,padding='same',activation=None)(seq)
    seq = BatchNormalization(axis=1)(seq)
    seq = ReLU()(seq)
    seq = SeparableConv1D(128,3,padding='same',activation=None)(seq)
    seq = BatchNormalization(axis=1)(seq)
    seq = ReLU()(seq)
    seq = MaxPooling1D(pool_size=2,strides=2)(seq)

    seq = SeparableConv1D(256,3,padding='same',activation=None)(seq)
    seq = BatchNormalization(axis=1)(seq)
    seq = ReLU()(seq)
    seq = SeparableConv1D(256,3,padding='same',activation=None)(seq)
    seq = BatchNormalization(axis=1)(seq)
    seq = ReLU()(seq)
    # seq = SeparableConv1D(256,3,padding='same',activation=None)(seq)
    seq = MaxPooling1D(pool_size=2,strides=2)(seq)

    seq = SeparableConv1D(256,3,padding='same',activation=None)(seq)
    seq = BatchNormalization(axis=1)(seq)
    seq = ReLU()(seq)
    seq = SeparableConv1D(256,3,padding='same',activation=None)(seq)
    seq = BatchNormalization(axis=1)(seq)
    seq = ReLU()(seq)
    # seq = SeparableConv1D(256,3,padding='same',activation=None)(seq)
    seq = MaxPooling1D(pool_size=2,strides=2)(seq)

    seq = SeparableConv1D(512,3,padding='same',activation=None)(seq)
    seq = BatchNormalization(axis=1)(seq)
    seq = ReLU()(seq)
    seq = SeparableConv1D(512,3,padding='same',activation=None)(seq)
    seq = BatchNormalization(axis=1)(seq)
    seq = ReLU()(seq)
    # seq = SeparableConv1D(512,3,padding='same',activation=None)(seq)
    seq = MaxPooling1D(pool_size=2,strides=2)(seq)

    seq = SeparableConv1D(512,3,padding='same',activation=None)(seq)
    seq = BatchNormalization(axis=1)(seq)
    seq = ReLU()(seq)
    seq = SeparableConv1D(512,3,padding='same',activation=None)(seq)
    seq = BatchNormalization(axis=1)(seq)
    seq = ReLU()(seq)
    # seq = SeparableConv1D(512,3,padding='same',activation='relu')(seq)
    seq = GlobalMaxPooling1D()(seq)

    # seq = Flatten()(seq)
    # seq = Dense(1000,activation='relu')(seq)

    seq = Dense(1000,activation='relu')(seq)

    output_tensor = Dense(16, activation='softmax')(seq)

    # model = Model(inputs=[input_tensor], outputs=[output_tensor])

    model = Model(inputs=[input_tensor], outputs=[output_tensor])
    # model = multi_gpu_model(model,gpus=4)
    model.summary()

    if load_weight==True:
        model.load_weights(weight_path,by_name=True)
    else:
        pass

    from keras.optimizers import Adam
    model.compile(loss='categorical_crossentropy',  # 交叉熵作为loss
                  optimizer=Adam(lr),
                  metrics=['accuracy'])

    if evalONtest == True:
        test_model = Model(inputs=[input_tensor], outputs=[output_tensor])

        test_model.compile(loss='categorical_crossentropy',  # 交叉熵作为loss
                           optimizer=Adam(lr),
                           metrics=['accuracy'])
        CSV_FILE_PATH2 = testFilePath
        data_to_test = load_data(CSV_FILE_PATH2)


        # train_x2, test_x2, train_y2, test_y2, Class_dict2
        # train_x2 = np.expand_dims(train_x2, axis=2)
        # test_x2 = np.expand_dims(test_x2, axis=2)

    else:
        pass

    print('开始加载数据')
    data_to_train = load_data(trainFilePath)
    # data_to_train = data_to_train[:10000]
    label_to_train = cut_letter(key_file,which_line,which_letter,key_length)
    label_to_train_lb = to_categorical(label_to_train,16)


    train_x,test_x,train_y,test_y = train_test_split(data_to_train,label_to_train_lb,test_size=test_size,shuffle=True)

    train_x = np.expand_dims(train_x, axis=2)
    test_x = np.expand_dims(test_x, axis=2)


    b_size = batch_size
    max_epochs = epochs
    print("Starting training ")

    learnratedecay = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, verbose=1, mode='auto',
                                       epsilon=0.0001, cooldown=0, min_lr=0)
    os.makedirs('/data/wuchenxi/allmodel/new_simeck_model/'+name+'/model',exist_ok=True)
    os.makedirs('/data/wuchenxi/allmodel/new_simeck_model/'+name+'/csvlog',exist_ok=True)
    os.makedirs('/data/wuchenxi/allmodel/new_simeck_model/'+name+'/tensorboard',exist_ok=True)

    checkpointer = ModelCheckpoint(monitor='val_loss',
                                   filepath='/data/wuchenxi/allmodel/new_simeck_model/'+name+'/model/' + name + '.hdf5',
                                   verbose=1, save_best_only=True)
    picture_output = TensorBoard(
        log_dir='/data/wuchenxi/allmodel/new_simeck_model/'+name+'/tensorboard/' + name + '_log',
        histogram_freq=0,
        write_graph=True,
        write_grads=True,
        write_images=True, )
    csvlog = CSVLogger(filename='/data/wuchenxi/allmodel/new_simeck_model/'+name+'/csvlog/' + name + '.csv',
                       separator=',', append=False)


    if evalONtest == True:
        pass
        # callback = [checkpointer, picture_output, csvlog, learnratedecay,
        #             EvaluateInputTensor(test_model, train_x2, train_y2,
        #                                 '/data/wuchenxi/allmodel/simeck_key_model/'+name+'/csvlog/' + name + '_test.csv')]
    else:
        callback = [checkpointer, picture_output, csvlog, learnratedecay]
    h = model.fit(train_x, train_y, batch_size=b_size, epochs=max_epochs, validation_data=(test_x, test_y),
                  shuffle=True, verbose=1,callbacks=callback)


# vgg16_model('/data/wuchenxi/new_simeck_data/signal15000train_val_test/train/',key_file='/data/wuchenxi/new_simeck_data/signal15000train_val_test/train_key.txt',
#             which_line=1,which_letter=2,key_length=4*10401,
#             testFilePath=None,
#             test_size=0.1,
#              batch_size=16,epochs=250,name='10401_1_2_v1_vgg16_1e-3',#'6000_7.16_multlabel_changehwplace'
#              lr=1e-3,evalONtest=False,
#              load_weight=False)

vgg16_model('/data/wuchenxi/new_simeck_data/signal54400_v2/signal54080_v2/',key_file='/data/wuchenxi/new_simeck_data/signal54400_v2/new_simeck_54080.txt',
            which_line=1,which_letter=1,key_length=4*54080,
            test_size=0.1,
            testFilePath=None,
             batch_size=16,epochs=250,name='54080_v2_1_1_v1',#'6000_7.16_multlabel_changehwplace'
             lr=1e-4,evalONtest=False,
             load_weight=False)


