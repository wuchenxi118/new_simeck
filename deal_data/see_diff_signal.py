from cut_label import cut_letter
from load_data import load_data

data_to_train = load_data('/data/wuchenxi/simeck_only_key/signal2019110702/')
label_to_train = cut_letter('/data/wuchenxi/simeck_only_key/only_collect_key_3200.txt',
                            0, 0, 4*3200)

for i,key in enumerate(label_to_train):
