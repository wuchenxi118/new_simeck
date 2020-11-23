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
# from deal_data.generate_load_data_SignalAddLabel import get_three_generater
from deal_data.shared_label_generate_load_data import get_three_generater
from keras.utils import multi_gpu_model

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

def resnet_model(trainFilePath,testFilePath,batch_size,epochs,name,lr,key_file,which_line,which_letter,load_weight=False,weight_path = None,evalONtest = True,validation_proportion=0.1,test_num = 100,rm_already_folder = True):
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
    share_bottom = MaxPooling1D(2)(seq)

    seq0 = BLOCK(share_bottom, 512)
    seq0 = BatchNormalization(axis=1)(seq0)
    seq0 = MaxPooling1D(2)(seq0)
    seq0 = BLOCK(seq0, 512)
    seq0 = BatchNormalization(axis=1)(seq0)
    # seq0 = Dropout(0.1)(seq0)
    seq0 = GlobalMaxPooling1D()(seq0)
    output_tensor0 = Dense(16, activation='softmax',name='output_tensor0')(seq0)

    seq1 = BLOCK(share_bottom, 512)
    seq1 = BatchNormalization(axis=1)(seq1)
    seq1 = MaxPooling1D(2)(seq1)
    seq1 = BLOCK(seq1, 512)
    seq1 = BatchNormalization(axis=1)(seq1)
    # seq1 = Dropout(0.1)(seq1)
    seq1 = GlobalMaxPooling1D()(seq1)
    output_tensor1 = Dense(16, activation='softmax',name='output_tensor1')(seq1)

    seq2 = BLOCK(share_bottom, 512)
    seq2 = BatchNormalization(axis=1)(seq2)
    seq2 = MaxPooling1D(2)(seq2)
    seq2 = BLOCK(seq2, 512)
    seq2 = BatchNormalization(axis=1)(seq2)
    # seq2 = Dropout(0.1)(seq2)
    seq2 = GlobalMaxPooling1D()(seq2)
    output_tensor2 = Dense(16, activation='softmax',name='output_tensor2')(seq2)

    seq3 = BLOCK(share_bottom, 512)
    seq3 = BatchNormalization(axis=1)(seq3)
    seq3 = MaxPooling1D(2)(seq3)
    seq3 = BLOCK(seq3, 512)
    seq3 = BatchNormalization(axis=1)(seq3)
    # seq3 = Dropout(0.1)(seq3)
    seq3 = GlobalMaxPooling1D()(seq3)
    output_tensor3 = Dense(16, activation='softmax',name='output_tensor3')(seq3)

    model = Model(inputs=[input_tensor], outputs=[output_tensor0,output_tensor1,output_tensor2,output_tensor3])
    model = multi_gpu_model(model, gpus=2)
    model.summary()

    if load_weight==True:
        model.load_weights(weight_path,by_name=True)
    else:
        pass

    from keras.optimizers import Adam
    model.compile(loss=['categorical_crossentropy','categorical_crossentropy','categorical_crossentropy','categorical_crossentropy'],  # 交叉熵作为loss
                  optimizer=Adam(lr),
                  metrics=['accuracy'])

    if evalONtest == True:
        test_model = Model(inputs=[input_tensor], outputs=[output_tensor0,output_tensor1,output_tensor2,output_tensor3])

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
                            test_num=test_num,batch_size=b_size,rm_already_folder = rm_already_folder)


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
                   verbose=1,callbacks=callback,
                            validation_data=val_generater,validation_steps=step_per_epoch_default_val)


def resnet_model_v3(trainFilePath,testFilePath,batch_size,epochs,name,lr,key_file,which_line,which_letter,key_length = None,load_weight=False,weight_path = None,evalONtest = False,test_size = 0.1):
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
    share_bottom = MaxPooling1D(2)(seq)

    seq0 = BLOCK(share_bottom, 256)
    seq0 = BatchNormalization(axis=1)(seq0)
    seq0 = MaxPooling1D(2)(seq0)
    seq0 = BLOCK(seq0, 256)
    seq0 = BatchNormalization(axis=1)(seq0)
    seq0 = MaxPooling1D(2)(seq0)
    seq0 = BLOCK(seq0, 512)
    seq0 = BatchNormalization(axis=1)(seq0)
    seq0 = MaxPooling1D(2)(seq0)
    seq0 = BLOCK(seq0, 512)
    seq0 = BatchNormalization(axis=1)(seq0)
    # seq0 = Dropout(0.1)(seq0)
    seq0 = GlobalMaxPooling1D()(seq0)
    output_tensor0 = Dense(16, activation='softmax',name='output_tensor0')(seq0)


    seq1 = BLOCK(share_bottom, 256)
    seq1 = BatchNormalization(axis=1)(seq1)
    seq1 = MaxPooling1D(2)(seq1)
    seq1 = BLOCK(seq1, 256)
    seq1 = BatchNormalization(axis=1)(seq1)
    seq1 = MaxPooling1D(2)(seq1)
    seq1 = BLOCK(seq1, 512)
    seq1 = BatchNormalization(axis=1)(seq1)
    seq1 = MaxPooling1D(2)(seq1)
    seq1 = BLOCK(seq1, 512)
    seq1 = BatchNormalization(axis=1)(seq1)
    # seq1 = Dropout(0.1)(seq1)
    seq1 = GlobalMaxPooling1D()(seq1)
    output_tensor1 = Dense(16, activation='softmax',name='output_tensor1')(seq1)

    seq2 = BLOCK(share_bottom, 256)
    seq2 = BatchNormalization(axis=1)(seq2)
    seq2 = MaxPooling1D(2)(seq2)
    seq2 = BLOCK(seq2, 256)
    seq2 = BatchNormalization(axis=1)(seq2)
    seq2 = MaxPooling1D(2)(seq2)
    seq2 = BLOCK(seq2, 512)
    seq2 = BatchNormalization(axis=1)(seq2)
    seq2 = MaxPooling1D(2)(seq2)
    seq2 = BLOCK(seq2, 512)
    seq2 = BatchNormalization(axis=1)(seq2)
    # seq2 = Dropout(0.1)(seq2)
    seq2 = GlobalMaxPooling1D()(seq2)
    output_tensor2 = Dense(16, activation='softmax',name='output_tensor2')(seq2)

    seq3 = BLOCK(share_bottom, 256)
    seq3 = BatchNormalization(axis=1)(seq3)
    seq3 = MaxPooling1D(2)(seq3)
    seq3 = BLOCK(seq3, 256)
    seq3 = BatchNormalization(axis=1)(seq3)
    seq3 = MaxPooling1D(2)(seq3)
    seq3 = BLOCK(seq3, 512)
    seq3 = BatchNormalization(axis=1)(seq3)
    seq3 = MaxPooling1D(2)(seq3)
    seq3 = BLOCK(seq3, 512)
    seq3 = BatchNormalization(axis=1)(seq3)
    # seq3 = Dropout(0.1)(seq3)
    seq3 = GlobalMaxPooling1D()(seq3)
    output_tensor3 = Dense(16, activation='softmax',name='output_tensor3')(seq3)

    model = Model(inputs=[input_tensor], outputs=[output_tensor0,output_tensor1,output_tensor2,output_tensor3])
    model = multi_gpu_model(model, gpus=2)
    model.summary()

    if load_weight==True:
        model.load_weights(weight_path,by_name=True)
    else:
        pass

    from keras.optimizers import Adam
    model.compile(loss=['categorical_crossentropy','categorical_crossentropy','categorical_crossentropy','categorical_crossentropy'],  # 交叉熵作为loss
                  optimizer=Adam(lr),
                  metrics=['accuracy'])

    if evalONtest == True:
        test_model = Model(inputs=[input_tensor], outputs=[output_tensor0,output_tensor1,output_tensor2,output_tensor3])

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
    label_to_train0 = cut_letter(key_file,which_line,which_letter,key_length)
    label_to_train_lb0 = to_categorical(label_to_train0,16)
    label_to_train1 = cut_letter(key_file,which_line,which_letter+1,key_length)
    label_to_train_lb1 = to_categorical(label_to_train1,16)
    label_to_train2 = cut_letter(key_file,which_line,which_letter+2,key_length)
    label_to_train_lb2 = to_categorical(label_to_train2,16)
    label_to_train3 = cut_letter(key_file,which_line,which_letter+3,key_length)
    label_to_train_lb3 = to_categorical(label_to_train3,16)

    train_x,test_x,train_y0,test_y0,train_y1,test_y1,train_y2,test_y2,train_y3,test_y3, = train_test_split(data_to_train,label_to_train_lb0,label_to_train_lb1,label_to_train_lb2,
                                                     label_to_train_lb3,test_size=test_size,shuffle=True)

    train_x = np.expand_dims(train_x, axis=2)
    test_x = np.expand_dims(test_x, axis=2)




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


    h = model.fit(train_x,[train_y0,train_y1,train_y2,train_y3],
                             epochs=epochs,
                   verbose=1,callbacks=callback,shuffle=True,
                            validation_data= (test_x, [test_y0,test_y1,test_y2,test_y3]),batch_size=batch_size,
                            )

    del h, train_x, train_y0, test_x, test_y0,train_y1,train_y2,train_y3,test_y1,test_y2,test_y3
    gc.collect()


resnet_model_v3('/data/wuchenxi/new_simeck_data/signal108800_circle/signal108800/',
             key_file='/data/wuchenxi/new_simeck_data/signal108800_circle/new_key_108800.txt',
            which_line=0,which_letter=0,key_length=4*108800,
            testFilePath=None,
             batch_size=16,epochs=50,name='108800_0_0123_hardshare_v3(no_drop)_circle_full_dataset_withshuffle',#'6000_7.16_multlabel_changehwplace'
             lr=1e-4,evalONtest=False,
             load_weight=False,
             weight_path='/data/wuchenxi/allmodel/new_simeck_model/108800_0_0123_hardshare_v1_circle_generatermode/model/108800_0_0123_hardshare_v1_circle_generatermode.hdf5',
            )