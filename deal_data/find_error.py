# import cv2
import matplotlib as mat
import sklearn
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import cluster, datasets

image_folder = '/data/wuchenxi/new_simeck_data/signal38400/'

image_list = os.listdir(image_folder)

image_list.sort()


wav = np.loadtxt(image_folder  +   'em_signal_7013.txt', delimiter=',', dtype='str')
wav = np.delete(wav, 35000)
wav = wav.astype(np.float64)

wav_v = sum(abs(wav))/35000
print(wav_v)

plt.plot(wav)
plt.show()


#['em_signal_18849.txt', 'em_signal_38401.txt', 'em_signal_38403.txt', 'em_signal_38404.txt', 'em_signal_7013.txt']

image_data = []
mean_list = []
nsum=0


# for i in image_list:
#
#     wav = np.loadtxt(image_folder + i, delimiter=',', dtype='str')
#     wav = np.delete(wav, 35000)
#     wav = wav.astype(np.float64)
#     wav_v = sum(abs(wav)) / 35000
#     mean_list.append(wav_v)
#
#     nsum+=wav_v
#
#
#
#
#     if wav_v>0.009:
#         image_data.append(i)
#
#
# nsum=nsum/38400
#
#
#
# print(nsum)
#
# print(mean_list)
#
# print(image_data)
