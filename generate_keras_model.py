from keras.metrics import top_k_categorical_accuracy
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
from keras.callbacks import Callback
import os
from sklearn.metrics import confusion_matrix,f1_score,roc_curve,auc
import matplotlib.pyplot as plt
from deal_data.cut_label import cut_letter
from deal_data.load_data import load_data
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
from deal_data.generate_load_data import get_three_generater

def BLOCK(seq, filters): # 定义网络的Block
    cnn = Conv1D(filters*2, 3, padding='SAME', dilation_rate=1, activation='relu')(seq)
    cnn = BatchNormalization(axis=1)(cnn)
    cnn = Lambda(lambda x: x[:,:,:filters] + x[:,:,filters:])(cnn)
    cnn = Conv1D(filters*2, 3, padding='SAME', dilation_rate=2, activation='relu')(cnn)
    cnn = BatchNormalization(axis=1)(cnn)
    cnn = Lambda(lambda x: x[:,:,:filters] + x[:,:,filters:])(cnn)
    cnn = Conv1D(filters*2, 3, padding='SAME', dilation_rate=4, activation='relu')(cnn)
    cnn = BatchNormalization(axis=1)(cnn)
    cnn = Lambda(lambda x: x[:,:,:filters] + x[:,:,filters:])(cnn)
    if int(seq.shape[-1]) != filters:
        seq = Conv1D(filters, 1, padding='SAME')(seq)
    seq = add([seq, cnn])
    return seq

def resnet_model(trainFilePath,testFilePath,batch_size,epochs,name,lr,key_file,which_line,which_letter,load_weight=False,weight_path = None,evalONtest = True,validation_proportion=0.3,test_num = 100):
    input_tensor = Input(shape=(35000, 1))
    seq = input_tensor
    seq = BLOCK(seq, 64)
    seq = BatchNormalization(axis=1)(seq)
    seq = MaxPooling1D(2)(seq)

    seq = BLOCK(seq, 64)
    seq = BatchNormalization(axis=1)(seq)
    seq = MaxPooling1D(2)(seq)

    seq = BLOCK(seq, 128)
    seq = BatchNormalization(axis=1)(seq)
    seq = MaxPooling1D(2)(seq)

    seq = BLOCK(seq, 128)
    seq = BatchNormalization(axis=1)(seq)
    seq = MaxPooling1D(2)(seq)

    seq = BLOCK(seq, 256)
    seq = BatchNormalization(axis=1)(seq)
    seq = MaxPooling1D(2)(seq)

    seq = BLOCK(seq, 256)
    seq = BatchNormalization(axis=1)(seq)
    seq = MaxPooling1D(2)(seq)

    seq = BLOCK(seq, 512)
    seq = BatchNormalization(axis=1)(seq)
    seq = MaxPooling1D(2)(seq)

    seq = BLOCK(seq, 512)
    seq = BatchNormalization(axis=1)(seq)

    seq = Dropout(0.6)(seq)

    seq = GlobalMaxPooling1D()(seq)

    output_tensor = Dense(16, activation='softmax')(seq)

    model = Model(inputs=[input_tensor], outputs=[output_tensor])
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
    b_size = batch_size
    max_epochs = epochs

    train_generater,val_generater,test_generater,train_data_num,val_data_num = \
        get_three_generater(trainFilePath,key_file,which_line,which_letter,
                            validation_proportion=validation_proportion,
                            test_num=test_num,batch_size=b_size)


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

    step_per_epoch_default = int(train_data_num/batch_size)
    step_per_epoch_default_val = int(val_data_num/batch_size)
    h = model.fit_generator(train_generater,
                            steps_per_epoch=step_per_epoch_default, epochs=max_epochs,
                   verbose=1,callbacks=callback,workers=4,use_multiprocessing=True,
                            validation_data=val_generater,validation_steps=step_per_epoch_default_val)


resnet_model('/data/wuchenxi/new_simeck_data/signal15000',key_file='/data/wuchenxi/new_simeck_data/new_simeck_15000.txt',
            which_line=1,which_letter=1,
            testFilePath=None,
             batch_size=16,epochs=250,name='15000_1_1_v1_generatermode',#'6000_7.16_multlabel_changehwplace'
             lr=1e-4,evalONtest=False,
             load_weight=False,weight_path='/data/wuchenxi/allmodel/new_simeck_model/10000_1_1_v1/model/6000_1_1_v1.hdf5')

