import os
import shutil

ori_folder = '/data/wuchenxi/new_simeck_data/signal108800_circle/signal108800/'
dst_folder = '/data/wuchenxi/new_simeck_data/signal108800_circle/signal54400/'

ori_list = os.listdir(ori_folder)

ori_list.sort(key = lambda x:int(x[10:-4]))

# print(ori_list[:54400])

for i in ori_list[:54400]:
    shutil.copyfile(ori_folder+i,dst_folder+i)