import numpy as np
import os
import shutil
from joblib import Parallel, delayed
from tqdm import tqdm
from deal_data.cut_label import cut_letter
from keras.utils import to_categorical
import random



def get_three_generater(dataset_path,key_path,which_line,which_letter,batch_size,validation_proportion=0.1,test_num = 320):
    """

    :param dataset_path:
    :param key_path:
    :param which_line:
    :param which_letter:
    :param validation_proportion:
    :param test_num:
    :return: three generater   train val and test
    """

    three_package = divided_dataset_to_train_and_validation(dataset_path,key_path,validation_proportion,test_num)
    train_data_path, train_label_path = three_package[0][0], three_package[0][1]
    val_data_path, val_label_path = three_package[1][0], three_package[1][1]
    test_data_path, test_label_path = three_package[2][0], three_package[2][1]

    train_key_length = int(len(os.listdir(train_data_path))*4)
    val_key_length = int(len(os.listdir(val_data_path))*4)
    test_key_length = int(len(os.listdir(test_data_path))*4)

    train_generater = generate_load_data(train_data_path, train_label_path, which_line, which_letter, train_key_length,batch_size=batch_size)
    val_generater = generate_load_data(val_data_path, val_label_path,which_line,which_letter,val_key_length,batch_size=batch_size)
    test_generater = generate_load_data(test_data_path, test_label_path,which_line,which_letter,test_key_length,batch_size=batch_size)
    train_data_num = int(train_key_length/4)
    val_data_num = int(val_key_length / 4)
    return train_generater,val_generater,test_generater,train_data_num,val_data_num


def divided_dataset_to_train_and_validation(dataset_path,key_path,validation_proportion=0.1,test_num = 320):
    """
    :param dataset_path: datapath no '/' at last
    :param key_path:  key_file path
    :param validation_proportion:  how many validation signal in dataset   this is a proportion
    :param test_num: how many test signal in dataset    this is a int number
    :return: the path of three dataset and keyfile
    """
    signal_list = os.listdir(dataset_path)
    signal_list.sort(key=lambda x: int(x[10:-4]))

    """to assert if data and label number are equal"""
    signal_num = len(signal_list)
    line_count=0
    for line in open(key_path):
        line_count+=1
    line_count = int(line_count/4)
    print(line_count,signal_num)
    assert line_count==signal_num

    """to mkdir the train validation and test signal to saved"""
    upper_dir = os.path.dirname(dataset_path)
    dataset_name = dataset_path.split('/')[-1]
    all_dir = upper_dir + '/'+dataset_name+'train_val_test'
    train_dir = upper_dir + '/'+dataset_name+'train_val_test' + '/train'
    val_dir = upper_dir + '/'+dataset_name+'train_val_test' + '/val'
    test_dir = upper_dir + '/'+dataset_name+'train_val_test' + '/test'
    if os.path.exists(all_dir):
        shutil.rmtree(all_dir)
    os.mkdir(all_dir)
    os.mkdir(train_dir)
    os.mkdir(val_dir)
    os.mkdir(test_dir)

    """begin to move signal and key"""
    key_file = open(key_path,'r')
    key_file = list(key_file)


    test_label_dir = all_dir + '/' + 'test_key.txt'
    val_label_dir = all_dir + '/' + 'val_key.txt'
    train_label_dir = all_dir + '/' + 'train_key.txt'

    test_label_file = open(test_label_dir, 'w')
    val_label_file = open(val_label_dir, 'w')
    train_label_file = open(train_label_dir, 'w')


    key_indx=0

    signal_key_pair_list = []
    for indx, signal in enumerate(signal_list):
        signal_key_pair = [signal,key_file[key_indx:key_indx+4]]
        signal_key_pair_list.append(signal_key_pair)
        key_indx += 4



    val_number = int((signal_num-test_num) * validation_proportion)
    # print(val_number)


    test_signal_key_pair_list = signal_key_pair_list[:test_num]
    res_signal_key_pair_list = signal_key_pair_list[test_num:]

    random.shuffle(res_signal_key_pair_list)

    val_signal_key_pair_list = res_signal_key_pair_list[:val_number]
    train_signal_key_pair_list = res_signal_key_pair_list[val_number:]


    # print(len(train_signal_key_pair_list))
    # print(train_signal_key_pair_list)


    for test_signal,test_key in test_signal_key_pair_list:
        shutil.copyfile(src=dataset_path + '/' + test_signal,
                            dst= test_dir + '/' + test_signal)
        test_label_file.writelines(test_key[0])
        test_label_file.writelines(test_key[1])
        test_label_file.writelines(test_key[2])
        test_label_file.writelines(test_key[3])

    for val_signal,val_key in val_signal_key_pair_list:
        shutil.copyfile(src=dataset_path + '/' + val_signal,
                            dst= val_dir + '/' + val_signal)
        val_label_file.writelines(val_key[0])
        val_label_file.writelines(val_key[1])
        val_label_file.writelines(val_key[2])
        val_label_file.writelines(val_key[3])

    for train_signal,train_key in train_signal_key_pair_list:
        shutil.copyfile(src=dataset_path + '/' + train_signal,
                            dst= train_dir + '/' + train_signal)
        train_label_file.writelines(train_key[0])
        train_label_file.writelines(train_key[1])
        train_label_file.writelines(train_key[2])
        train_label_file.writelines(train_key[3])






    return [(train_dir,train_label_dir), (val_dir,val_label_dir),(test_dir,test_label_dir)]




    # for line in open(key_path):


def generate_load_data(signal_data_path,key_path,which_line,which_letter,key_length,batch_size):
    # def parallel_load(signalname, signal_data_path):
    #     nonlocal signal_nparray
    #     wav = np.loadtxt(signal_data_path + signalname, delimiter=',', dtype='str')
    #     wav = np.delete(wav, 10000)
    #     wav = wav.astype(np.float32)
    #     signal_nparray.append(wav)
    """

    :param signal_data_path: path of signal folder
    :return: all signal numpy array
    """
    while True:
        signal_list = os.listdir(signal_data_path)
        print(signal_list)
        # signal_list.sort(key = lambda x:int(x[10:-4]))
        key_label = cut_letter(key_path,which_line,which_letter,key_length)
        # key_label_lb = to_categorical(key_label,16)

        if len(key_label)==len(signal_list):
            pass
        else:
            print(len(key_label),len(signal_list))
            raise ValueError

        batch_size = batch_size
        indx_to_count = 0
        train_x_batch = np.zeros((batch_size,35000,1))
        train_y_batch = np.zeros((batch_size,16))
        for indx,signal in enumerate(signal_list):

            wav = np.loadtxt(signal_data_path +'/'+ signal, delimiter=',', dtype='str')
            wav = np.delete(wav, 35000)
            wav = wav.astype(np.float32)
            key_label_lb = to_categorical(key_label[indx], 16)
            key_label_lb = np.expand_dims(key_label_lb,axis=0)
            """
            must use np.expand_dims(key_label_lb,axis=0)  to change the data.shape to (1,16)  from (16,)
            """
            # print(key_label_lb.shape)
            train_x = np.expand_dims(wav, axis=2)
            train_x = np.expand_dims(train_x, axis=0)

            train_x_batch[indx_to_count]=train_x
            train_y_batch[indx_to_count]=key_label_lb



            if indx_to_count == batch_size-1:
                # print(train_x_batch.shape)
                # print(train_y_batch.shape)
                yield ({'input_1': train_x_batch}, {'dense_1': train_y_batch})
                train_x_batch = np.zeros((batch_size, 35000, 1))
                train_y_batch = np.zeros((batch_size, 16))
                indx_to_count=-1

            indx_to_count += 1















if __name__=='__main__':

    # g = generate_load_data(signal_data_path='/data/wuchenxi/new_simeck_data/signal13533/',
    #                    key_path='/data/wuchenxi/new_simeck_data/new_simeck_13533.txt',
    #                    which_line=0,which_letter=0,key_length=4*13533,batch_size=16)
    #
    # print(next(g))
    # print(next(g))
    # print(next(g))
    # divided_dataset_to_train_and_validation('/data/wuchenxi/new_simeck_data/signal54400_circle/signal54080circle',key_path='/data/wuchenxi/new_simeck_data/signal54400_circle/new_simeck_54080.txt')


    # get_three_generater(dataset_path='/data/wuchenxi/new_simeck_data/signal54400_circle/signal54080circle',
    #                     key_path='/data/wuchenxi/new_simeck_data/signal54400_v2/new_simeck_54080.txt',
    #                     which_line=0,which_letter=1,batch_size=16)

    g = generate_load_data(signal_data_path='/data/wuchenxi/new_simeck_data/signal54400_circle/signal54080circletrain_val_test/train',
                       key_path='/data/wuchenxi/new_simeck_data/signal54400_circle/signal54080circletrain_val_test/train_key.txt',
                       which_line=0, which_letter=1, key_length=48384*4, batch_size=1)
    print(next(g))



