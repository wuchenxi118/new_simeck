from keras_model import resnet_model
from simple_keras_model import vgg16_model



def train_48bit_model(input_key_list,model_name,
                      trainFilePath,testFilePath,batch_size,epochs,lr,key_file,
                      key_length,test_size,which_model,load_weight=False,weight_path = None,evalONtest = True):

    if which_model=='resnet':
        for i in input_key_list:

            resnet_model(trainFilePath=trainFilePath,testFilePath=testFilePath,batch_size=batch_size,
                         epochs=epochs,name=model_name+'_'+str(i[0])+'_'+str(i[1])+'_v2(trfmerge_drop(0.1)).hdf5',
                         lr=lr,key_file=key_file,which_line=i[0],which_letter=i[1],key_length=key_length,
                         test_size=test_size,load_weight=load_weight,weight_path=weight_path,evalONtest=evalONtest)
    if which_model=='vgg16':
        for i in input_key_list:
            vgg16_model(trainFilePath=trainFilePath, testFilePath=testFilePath, batch_size=batch_size,
                         epochs=epochs,
                         name=model_name + '_' + str(i[0]) + '_' + str(i[1]) + '_v1.hdf5',
                         lr=lr, key_file=key_file, which_line=i[0], which_letter=i[1], key_length=key_length,
                         test_size=test_size, load_weight=load_weight, weight_path=weight_path, evalONtest=evalONtest)

















if __name__ == '__main__':

    # input_key = [[1,0],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]]
    input_key = [[0,0],[0,1],[0,2],[0,3]]
    # input_key = [[3,3]]
    train_48bit_model(input_key_list=input_key,model_name='signal9690_circle_trans_fromMergenoise80000',
                      trainFilePath='/data/wuchenxi/new_simeck_data/signal54400_circle/signal321_10000/',
                      testFilePath=None,batch_size=16,epochs=20,lr=1e-4,
                      key_file='/data/wuchenxi/new_simeck_data/signal54400_circle/new_simeck_321_10000.txt',
                      key_length=9680*4,test_size=0.1,load_weight=True,
                      weight_path='/data/wuchenxi/allmodel/new_simeck_model/mergenoise80000_temple_0_0/model/mergenoise80000_temple_0_0.hdf5',
                      evalONtest=False,
                      which_model='resnet')