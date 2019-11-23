import os



def cut_letter(key_file,which_line,which_letter,length):
    """

    :param key_file:  The path of the key_file
    :param which_line: which line of 4 line , the value can passed is 0-3
    :param which_letter: which letter of 4 line , the value can passed is 0-3
    :param length: how many signal in the key file ,if you have 3200 signals, this param is 3200*4
    :return: a key list,value is 0-15
    """

    label_list=[]
    i = 0
    filedir = key_file
    for line in open(filedir):
        if i% 4 ==which_line:
            letter_list = list(line)
            label_letter = letter_list[which_letter]
            if label_letter=='a':
                label_letter='10'
            if label_letter=='b':
                label_letter='11'
            if label_letter=='c':
                label_letter='12'
            if label_letter=='d':
                label_letter='13'
            if label_letter=='e':
                label_letter='14'
            if label_letter=='f':
                label_letter='15'
            label_list.append(label_letter)
        i+=1
        if i==length:
            break


    return label_list

if __name__=='__main__':

    c = cut_letter('/data/wuchenxi/new_simeck_data/signal13533train_val_test/test_key.txt',which_line=0,which_letter=0,length=4*100)
    print(c)
    print(len(c))
