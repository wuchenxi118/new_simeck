import os
import numpy as np
import pandas as pd
import random
from deal_data.load_data import load_data
from deal_data.cut_label import cut_letter
import gc
#from tqdm import tqdm
import time
from keras.utils import to_categorical



def add_one_class_in_mem(ori_data,target_list,
                         class_str,forloop_number_input,choose_number):

    target_list_class = [i for i, x in enumerate(target_list) if x == class_str]
    target_list_class_length = len(target_list_class)
    merge_data_list = []

    print(class_str,'length_is:',target_list_class_length)


    forloop_number = forloop_number_input  #each class has 2000 examples

    var_builder = locals()
    for create_var in range(forloop_number):
        var_builder['merge_data'+str(create_var)] = 0


    for numbers in range(forloop_number):

        if choose_number == 2:
            if numbers % 100 == 0:
                print('begin', ' merge_data', class_str, numbers)
            random_number1 = random.randint(0, target_list_class_length - 1)
            random_number2 = random.randint(0, target_list_class_length - 1)

            var_builder['merge_data' + str(numbers)] = \
                ori_data[target_list_class[random_number1], :] + ori_data[target_list_class[random_number2],:]

        if choose_number == 3:
            if numbers % 100 == 0:
                print('begin', ' merge_data', class_str, numbers)
            random_number1 = random.randint(0, target_list_class_length - 1)
            random_number2 = random.randint(0, target_list_class_length - 1)
            random_number3 = random.randint(0, target_list_class_length - 1)

            var_builder['merge_data' + str(numbers)] = \
                ori_data[target_list_class[random_number1], :] + ori_data[target_list_class[random_number2],:] \
                + ori_data[target_list_class[random_number3], :]


        if choose_number == 4:
            if numbers % 100 == 0:
                print('begin', ' merge_data', class_str, numbers)
            random_number1 = random.randint(0, target_list_class_length - 1)
            random_number2 = random.randint(0, target_list_class_length - 1)
            random_number3 = random.randint(0, target_list_class_length - 1)
            random_number4 = random.randint(0, target_list_class_length - 1)
            var_builder['merge_data' + str(numbers)] = \
                ori_data[target_list_class[random_number1], :] + ori_data[target_list_class[random_number2],:] \
                + ori_data[target_list_class[random_number3], :] + ori_data[target_list_class[random_number4],:]
        if choose_number == 10:
            if numbers % 100 == 0:
                print('begin', ' merge_data', class_str, numbers)
            random_number1 = random.randint(0, target_list_class_length - 1)
            random_number2 = random.randint(0, target_list_class_length - 1)
            random_number3 = random.randint(0, target_list_class_length - 1)
            random_number4 = random.randint(0, target_list_class_length - 1)
            random_number5 = random.randint(0, target_list_class_length - 1)
            random_number6 = random.randint(0, target_list_class_length - 1)
            random_number7 = random.randint(0, target_list_class_length - 1)
            random_number8 = random.randint(0, target_list_class_length - 1)
            random_number9 = random.randint(0, target_list_class_length - 1)
            random_number10 = random.randint(0, target_list_class_length - 1)
            var_builder['merge_data' + str(numbers)] = \
                ori_data[target_list_class[random_number1], :] + ori_data[target_list_class[random_number2],:] \
                + ori_data[target_list_class[random_number3], :] + ori_data[target_list_class[random_number4],:] \
                + ori_data[target_list_class[random_number5], :] + ori_data[target_list_class[random_number6],:] \
                + ori_data[target_list_class[random_number7], :] + ori_data[target_list_class[random_number8],:] \
                + ori_data[target_list_class[random_number9], :] + ori_data[target_list_class[random_number10], :]


        var_builder['merge_data' + str(numbers)] = (np.around(var_builder['merge_data'+str(numbers)],\
                                                                 decimals=6))/choose_number
        var_builder['merge_data' + str(numbers)] = var_builder['merge_data' + str(numbers)].astype(np.float32)

        merge_data_list.append(var_builder['merge_data' + str(numbers)])

    # del ori_data
    # gc.collect()

    merge_data_list = np.array(merge_data_list)

    return merge_data_list


def add_all_class_in_mem_return_ori_and_add_data(signal_data_path,label_path,which_line,which_letter,key_length,
                                                 each_class_number,choose_number,add_ori_data=False):
    ori_data = load_data(signal_data_path=signal_data_path)
    target_list = cut_letter(label_path, which_line=which_line,
                   which_letter=which_letter, length=key_length)

    class_list = set(target_list)
    class_list = [int(i) for i in class_list]
    class_list.sort()
    class_list = [str(i) for i in class_list]

    # class_list = [str(i) for i in range(16)]

    train_np_array = np.zeros((each_class_number*len(class_list), 35000))
    train_y = []

    np_index = 0
    # y_index = 0

    for str_index in class_list:
        class_array=add_one_class_in_mem(ori_data,target_list,class_str=str_index,forloop_number_input=each_class_number,choose_number=choose_number)
        train_np_array[np_index:np_index+each_class_number,:] = class_array
        np_index+=each_class_number

        class_list = [int(str_index)]*each_class_number
        for single_label in class_list:
            train_y.append(single_label)
        # y_index+=1

    if add_ori_data == True:
        train_np_array = np.concatenate((train_np_array,ori_data),axis=0)
        del ori_data
        gc.collect()
        for ori_label in target_list:
            train_y.append(ori_label)
    del ori_data
    gc.collect()

    train_y = to_categorical(train_y,16)




    return train_np_array,train_y




if __name__=='__main__':


    # data1 = add_one_class(class_str='10',forloop_number_input=200,class_save_path=None,choose_number=2)
    # data1 = add_one_class_in_mem(signal_data_path='/data/wuchenxi/new_simeck_data/signal54400_v2/signal321_10000/',
    #                              label_path='/data/wuchenxi/new_simeck_data/signal54400_v2/new_simeck_321_10000.txt',
    #                              which_line=0,which_letter=0,key_length=4*9680,
    #                              class_str='15',forloop_number_input=100,choose_number=10)
    data1,label1 = add_all_class_in_mem_return_ori_and_add_data(signal_data_path='/data/wuchenxi/new_simeck_data/signal54400_v2/signal321_10000/',
                                 label_path='/data/wuchenxi/new_simeck_data/signal54400_v2/new_simeck_321_10000.txt',
                                 which_line=0,which_letter=0,key_length=4*9680,
                                 each_class_number=200,choose_number=2,add_ori_data=True)
    print(data1.shape)
    print(label1.shape)
    # print(data1[0])
    print(data1)

    # do_it(each_class_number=200,save_path='/data/data1/data/wuchenxi/final/luoman_add4',c_number=4)