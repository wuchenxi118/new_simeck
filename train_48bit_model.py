from keras_model import resnet_model



def train_48bit_model(input_key_list,model_name,
                      trainFilePath,testFilePath,batch_size,epochs,lr,key_file,
                      key_length,test_size,load_weight=False,weight_path = None,evalONtest = True):
    for i in input_key_list:

        resnet_model(trainFilePath=trainFilePath,testFilePath=testFilePath,batch_size=batch_size,
                     epochs=epochs,name=model_name+'_'+str(i[0])+'_'+str(i[1])+'_v2(trfmerge_drop(0.1)).hdf5',
                     lr=lr,key_file=key_file,which_line=i[0],which_letter=i[1],key_length=key_length,
                     test_size=test_size,load_weight=load_weight,weight_path=weight_path,evalONtest=evalONtest)



















if __name__ == '__main__':

    # input_key = [[1,0],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]]
    input_key = [[1, 2],[2, 3],[3, 2],[3, 3]]
    train_48bit_model(input_key_list=input_key,model_name='321_10000_num9680_v2',
                      trainFilePath='/data/wuchenxi/new_simeck_data/signal54400_v2/signal321_10000/',
                      testFilePath=None,batch_size=16,epochs=50,lr=1e-4,
                      key_file='/data/wuchenxi/new_simeck_data/signal54400_v2/new_simeck_321_10000.txt',
                      key_length=9680*4,test_size=0.2,load_weight=True,
                      weight_path='/data/wuchenxi/allmodel/simeck_key_model/merge32000_1_4_4(v2untrans).hdf5',
                      evalONtest=False)