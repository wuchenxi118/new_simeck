import numpy as np
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from multiprocessing import Pool
import time



def load_single_data(signal):
    wav = np.loadtxt(signal, delimiter=',', dtype='str')
    wav = np.delete(wav, 35000)
    wav = wav.astype(np.float32)
    return wav


def load_data(signal_data_path):

    signal_list = os.listdir(signal_data_path)
    signal_list.sort(key = lambda x:int(x[10:-4]))

    signal_list=[signal_data_path+i for i in signal_list]


    # signal_nparray=[]

    pool = Pool(16)
    signal_nparray = pool.map(load_single_data,signal_list)


    signal_nparray = np.array(signal_nparray)

    return signal_nparray



if __name__=='__main__':

    # start = time.time()
    # data=load_data_old(signal_data_path='/data/wuchenxi/new_simeck_data/signal15000train_val_test/val/')
    # end = time.time()
    # print(str(end-start))
    # print(data.shape)

    start2 = time.time()
    data=load_data(signal_data_path='/data/wuchenxi/new_simeck_data/signal54400_v2/signal321_10000/')
    end2 = time.time()
    print(str(end2-start2))


#