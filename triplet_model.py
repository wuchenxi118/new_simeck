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
from keras.models import Model,load_model,Sequential
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,History,TensorBoard,CSVLogger
from keras.metrics import top_k_categorical_accuracy
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
from keras.callbacks import Callback
import os
from sklearn.metrics import confusion_matrix,f1_score,roc_curve,auc
import matplotlib.pyplot as plt
from keras.utils import multi_gpu_model
import gc
from deal_data.cut_label import cut_letter
from deal_data.load_data import load_data
from deal_data.add_data import add_all_class_in_mem_return_ori_and_add_data
import random
from itertools import permutations


def triplet_loss(y_true, y_pred, alpha=0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    print('y_pred.shape = ', y_pred)

    total_lenght = y_pred.shape.as_list()[-1]
    #     print('total_lenght=',  total_lenght)
    #     total_lenght =12

    anchor = y_pred[:, 0:int(total_lenght * 1 / 3)]
    positive = y_pred[:, int(total_lenght * 1 / 3):int(total_lenght * 2 / 3)]
    negative = y_pred[:, int(total_lenght * 2 / 3):int(total_lenght * 3 / 3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)

    return loss

def generate_triplet(x, y, testsize=0.3, ap_pairs=10, an_pairs=10):
    data_xy = tuple([x, y])


    trainsize = 1 - testsize

    triplet_train_pairs = []
    triplet_test_pairs = []

    data_classes = [int(i) for i in set(data_xy[1])]

    data_classes = [str(i) for i in sorted(data_classes)]

    for data_class in data_classes:

        same_class_idx = np.where((np.array(data_xy[1]) == data_class))[0]
        print(type(data_xy[1]))
        print(type(data_class))
        print(same_class_idx)
        diff_class_idx = np.where(np.array(data_xy[1]) != data_class)[0]
        A_P_pairs = random.sample(list(permutations(same_class_idx, 2)), k=ap_pairs)  # Generating Anchor-Positive pairs
        Neg_idx = random.sample(list(diff_class_idx), k=an_pairs)

        # train
        A_P_len = len(A_P_pairs)
        Neg_len = len(Neg_idx)
        for ap in A_P_pairs[:int(A_P_len * trainsize)]:
            Anchor = data_xy[0][ap[0]]
            Positive = data_xy[0][ap[1]]
            for n in Neg_idx:
                Negative = data_xy[0][n]
                triplet_train_pairs.append([Anchor, Positive, Negative])
                # test
        for ap in A_P_pairs[int(A_P_len * trainsize):]:
            Anchor = data_xy[0][ap[0]]
            Positive = data_xy[0][ap[1]]
            for n in Neg_idx:
                Negative = data_xy[0][n]
                triplet_test_pairs.append([Anchor, Positive, Negative])

    return np.array(triplet_train_pairs), np.array(triplet_test_pairs)


def create_base_network(in_dims):
    """
    Base network to be shared.
    """
    model = Sequential()
    model.add(Conv1D(128, 5, padding='same', input_shape=(35000,1), activation='relu',
                     name='conv1'))
    model.add(MaxPooling1D(2, padding='same', name='pool1'))
    model.add(Conv1D(256, 5, padding='same', activation='relu', name='conv2'))
    model.add(MaxPooling1D(2, padding='same', name='pool2'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(4, name='embeddings'))
    # model.add(Dense(600))

    return model

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

def resnet_model(trainFilePath,testFilePath,batch_size,epochs,name,lr,
                 key_file,which_line,which_letter,key_length,test_size,
                 use_add=False,each_class_number=None,choose_number=None,load_weight=False,weight_path = None,evalONtest = True):
    # input_tensor = Input(shape=(35000, 1))
    # seq = input_tensor
    # seq = BLOCK(seq, 64)
    # seq = BatchNormalization(axis=1)(seq)
    # seq = MaxPooling1D(2)(seq)
    #
    # seq = BLOCK(seq, 64)
    # seq = BatchNormalization(axis=1)(seq)
    # seq = MaxPooling1D(2)(seq)
    #
    # seq = BLOCK(seq, 128)
    # seq = BatchNormalization(axis=1)(seq)
    # seq = MaxPooling1D(2)(seq)
    #
    # seq = BLOCK(seq, 128)
    # seq = BatchNormalization(axis=1)(seq)
    # seq = MaxPooling1D(2)(seq)
    #
    # seq = BLOCK(seq, 256)
    # seq = BatchNormalization(axis=1)(seq)
    # seq = MaxPooling1D(2)(seq)
    #
    # seq = BLOCK(seq, 256)
    # seq = BatchNormalization(axis=1)(seq)
    # seq = MaxPooling1D(2)(seq)
    #
    # seq = BLOCK(seq, 512)
    # seq = BatchNormalization(axis=1)(seq)
    # seq = MaxPooling1D(2)(seq)
    #
    # seq = BLOCK(seq, 512)
    # seq = BatchNormalization(axis=1)(seq)
    #
    # # seq = Dropout(0.6)(seq)
    # seq = Dropout(0.1)(seq)
    #
    # seq = GlobalMaxPooling1D()(seq)
    #
    # output_tensor = Dense(16, activation='softmax')(seq)

    # model = Model(inputs=[input_tensor], outputs=[output_tensor])

    anchor_input = Input((35000, 1,), name='anchor_input')
    positive_input = Input((35000, 1,), name='positive_input')
    negative_input = Input((35000, 1,), name='negative_input')

    Shared_DNN = create_base_network([35000, 1])

    encoded_anchor = Shared_DNN(anchor_input)
    encoded_positive = Shared_DNN(positive_input)
    encoded_negative = Shared_DNN(negative_input)

    merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')
    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_vector)

    # model = Model(inputs=[input_tensor], outputs=[output_tensor])
    # model = multi_gpu_model(model,gpus=4)
    model.summary()

    if load_weight==True:
        model.load_weights(weight_path,by_name=True)
    else:
        pass

    from keras.optimizers import Adam
    model.compile(loss=triplet_loss,  # 交叉熵作为loss
                  optimizer=Adam(lr))

    # if evalONtest == True:
    #     test_model = Model(inputs=[input_tensor], outputs=[output_tensor])
    #
    #     test_model.compile(loss='categorical_crossentropy',  # 交叉熵作为loss
    #                        optimizer=Adam(lr),
    #                        metrics=['accuracy'])
    #     CSV_FILE_PATH2 = testFilePath
    #     data_to_test = load_data(CSV_FILE_PATH2)
    #
    #
    #     # train_x2, test_x2, train_y2, test_y2, Class_dict2
    #     # train_x2 = np.expand_dims(train_x2, axis=2)
    #     # test_x2 = np.expand_dims(test_x2, axis=2)
    #
    # else:
    #     pass

    print('开始加载数据')

    if use_add == True:
        data_to_train,label_to_train_lb = add_all_class_in_mem_return_ori_and_add_data(signal_data_path=trainFilePath,
                                                                                       label_path=key_file,
                                                                                       which_line=which_line,
                                                                                       which_letter=which_letter,
                                                                                       key_length=key_length,
                                                                                       each_class_number=each_class_number,
                                                                                       choose_number=choose_number)
    else:
        data_to_train = load_data(trainFilePath)
        # data_to_train = data_to_train[:10000]
        label_to_train = cut_letter(key_file,which_line,which_letter,key_length)
        label_to_train_lb = to_categorical(label_to_train,16)


    train_x,test_x,train_y,test_y = train_test_split(data_to_train,label_to_train,test_size=test_size,shuffle=True)

    train_x = np.expand_dims(train_x, axis=2)
    test_x = np.expand_dims(test_x, axis=2)

    X_train_triplet, X_test_triplet = generate_triplet(train_x, train_y, testsize=0.3, ap_pairs=10, an_pairs=10)
    print(X_train_triplet.shape)
    print(X_test_triplet.shape)
    # print(X_train_triplet)

    Anchor = X_train_triplet[:, 0,:, :].reshape(-1, 35000, 1)
    Positive = X_train_triplet[:, 1, :,:].reshape(-1, 35000, 1)
    Negative = X_train_triplet[:, 2,:, :].reshape(-1, 35000, 1)
    Anchor_test = X_test_triplet[:, 0, :,:].reshape(-1, 35000, 1)
    Positive_test = X_test_triplet[:, 1, :,:].reshape(-1, 35000, 1)
    Negative_test = X_test_triplet[:, 2, :,:].reshape(-1, 35000, 1)

    Y_dummy = np.empty((Anchor.shape[0], 300))
    Y_dummy2 = np.empty((Anchor_test.shape[0], 1))

    model.fit([Anchor, Positive, Negative], y=Y_dummy,
              validation_data=([Anchor_test, Positive_test, Negative_test], Y_dummy2), batch_size=512, epochs=500)

    print(Anchor.shape)


    # b_size = batch_size
    # max_epochs = epochs
    # print("Starting training ")
    #
    # learnratedecay = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, verbose=1, mode='auto',
    #                                    epsilon=0.0001, cooldown=0, min_lr=0)
    # os.makedirs('/data/wuchenxi/allmodel/new_simeck_model/'+name+'/model',exist_ok=True)
    # os.makedirs('/data/wuchenxi/allmodel/new_simeck_model/'+name+'/csvlog',exist_ok=True)
    # os.makedirs('/data/wuchenxi/allmodel/new_simeck_model/'+name+'/tensorboard',exist_ok=True)
    #
    # checkpointer = ModelCheckpoint(monitor='val_loss',
    #                                filepath='/data/wuchenxi/allmodel/new_simeck_model/'+name+'/model/' + name + '.hdf5',
    #                                verbose=1, save_best_only=True)
    # picture_output = TensorBoard(
    #     log_dir='/data/wuchenxi/allmodel/new_simeck_model/'+name+'/tensorboard/' + name + '_log',
    #     histogram_freq=0,
    #     write_graph=True,
    #     write_grads=True,
    #     write_images=True, )
    # csvlog = CSVLogger(filename='/data/wuchenxi/allmodel/new_simeck_model/'+name+'/csvlog/' + name + '.csv',
    #                    separator=',', append=False)
    #
    #
    # if evalONtest == True:
    #     pass
    #     # callback = [checkpointer, picture_output, csvlog, learnratedecay,
    #     #             EvaluateInputTensor(test_model, train_x2, train_y2,
    #     #                                 '/data/wuchenxi/allmodel/simeck_key_model/'+name+'/csvlog/' + name + '_test.csv')]
    # else:
    #     callback = [checkpointer, picture_output, csvlog, learnratedecay]
    # # h = model.fit(train_x, train_y, batch_size=b_size, epochs=max_epochs, validation_data=(test_x, test_y),
    # #               shuffle=True, verbose=1,callbacks=callback)
    #
    # del h,train_x,train_y,test_x,test_y
    # gc.collect()


# resnet_model('/data/wuchenxi/new_simeck_data/signal15000train_val_test/train/',key_file='/data/wuchenxi/new_simeck_data/signal15000train_val_test/train_key.txt',
#             which_line=1,which_letter=2,key_length=4*10401,
#             testFilePath=None,
#              batch_size=16,epochs=250,name='10401_1_2_v1',#'6000_7.16_multlabel_changehwplace'
#              lr=1e-4,evalONtest=False,
#              load_weight=True,weight_path='/data/wuchenxi/allmodel/new_simeck_model/54080_1_1_v1/model/54080_1_1_v1.hdf5')

#weight_path = '/data/wuchenxi/allmodel/simeck_key_model/merge32000_1_4_4(v2untrans).hdf5'

if __name__ == '__main__':
    resnet_model('/data/wuchenxi/new_simeck_data/signal54400_circle/signal_320_circle/',
                 key_file='/data/wuchenxi/new_simeck_data/signal54400_circle/new_simeck_320.txt',
                which_line=0,which_letter=3,key_length=4*320,
                test_size=0.1,
                testFilePath=None,
                 batch_size=16,epochs=30,name='test',#'6000_7.16_multlabel_changehwplace'
                 lr=1e-4,evalONtest=False,
                 use_add=False,each_class_number=3375,choose_number=2,
                 load_weight=True,weight_path='/data/wuchenxi/allmodel/new_simeck_model/54080_circle_v1_0_0/model/54080_circle_v1_0_0.hdf5')



