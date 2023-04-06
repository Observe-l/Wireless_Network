import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
import pandas as pd
# from tensorflow.keras import layers
import pickle
from sklearn import preprocessing
import mlflow
import optparse

# from udp_req import udp_send, udp_server


MODEL_PORT = 10652
DATA_PORT = 10653

def get_options():
    optParse = optparse.OptionParser()
    optParse.add_option("-l","--loss",default="0",type=str,help="loss_rate")
    optParse.add_option("-m","--model",default="0",type=int,help="model type")
    optParse.add_option("-a","--algo",default="0",type=int,help="algorithm")
    optParse.add_option("-f","--file",default="FD001",type=str,help="trainning dataset")
    options, args = optParse.parse_args()
    return options
options = get_options()

FILE_NAME = "re_train.txt"
DATA_TMP = "re_rec.txt"
ORIGIN_DATA = 'CMAPSSData/train_FD001.txt'
TEST_DATA = "CMAPSSData/test_" + options.file + ".txt"
RUL_FILE = "CMAPSSData/RUL_" + options.file + ".txt"
TIME_OUT = 3
REMAIN_NUM = 35

def reload_model(model_name:str, model_version:str = None) -> mlflow.pyfunc.PyFuncModel:
    '''
    The models are stored in the MLflow tracking server.
    Fetch the latest model
    '''
    if model_version:
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )
    else:
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/Production"
        )
    return model


# function to reshape features into seq_array: (samples, time steps, features)
def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means
    we need to drop those which are below the window-length. """
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]

# function to generate label_array
def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]

def data_load(file_name:str = FILE_NAME, sequence_length=25, model_test:bool = False):
    # Read data from txt file
    train_df = pd.read_csv(file_name, sep=" ",header=None)
    # print('latest machine: ',np.sort(train_df[0].unique())[::1][0])
    train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
    train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                        's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                        's15', 's16', 's17', 's18', 's19', 's20', 's21']
    # Generate labels. If RUL < trashold, label = 1, o.w 0
    rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    if model_test:
        truth_df = pd.read_csv(RUL_FILE, sep=" ", header=None)
        truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)
        truth_df.columns = ['more']
        truth_df['id'] = truth_df.index + 1
        truth_df['max'] = rul['max'] + truth_df['more']
        truth_df.drop('more', axis=1, inplace=True)
    else:
        truth_df = rul

    train_df = train_df.merge(truth_df, on=['id'], how='left')

    train_df['RUL'] = train_df['max'] - train_df['cycle']
    train_df.loc[train_df['RUL']>=130,'RUL'] = 130
    train_df.drop('max', axis=1, inplace=True)

    # generate label1 column for training data
    train_df['label1'] = np.where(train_df['RUL'] <= 30, 1, 0 )
    # Normalize the data
    scaler = preprocessing.MinMaxScaler()
    scaler_column = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                        's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                        's15', 's16', 's17', 's18', 's19', 's20', 's21']
    for col in scaler_column:
        train_df[[col]] = scaler.fit_transform(train_df[[col]])
    
    # pick the feature columns 
    sensor_cols = ['s' + str(i) for i in range(1,22)]
    sequence_cols = ['setting1', 'setting2', 'setting3','cycle']
    sequence_cols.extend(sensor_cols)
    # generator for the training sequences
    seq_gen = (list(gen_sequence(train_df[train_df['id']==id], sequence_length, sequence_cols)) 
            for id in train_df['id'].unique())
    # generate sequences and convert to numpy array
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
    # generate labels (generated from "label1" col as its binary classification)
    label_gen = [gen_labels(train_df[train_df['id']==id], sequence_length, ['RUL']) 
                for id in train_df['id'].unique()]
    label_array = np.concatenate(label_gen).astype(np.float32)
    return seq_array, label_array

def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    
    client = mlflow.tracking.MlflowClient()
    test_data, test_label = data_load(file_name=TEST_DATA,model_test=True)
    # loss_rate = ["loss0_","loss5_","loss10_","loss15_","loss20_","loss25_","loss30_","loss35_","loss40_","loss45_","loss50_","loss55_","loss60_","loss65_","loss70_","loss75_"]
    model_type = ["classification_","regression_"]
    algo_type = ["dnn","cnn","lstm","bilstm"]

    model_name = "loss" + options.loss + "_" + model_type[options.model] + algo_type[options.algo]
    result = []
    model_version = int(client.get_latest_versions(name=model_name,stages=['Production'])[0].version)
    result_file = "result/"+ options.file + "/" + model_name + ".p"
    print(model_name)
    for tmp_version in range(1,model_version+1):
        # model_dic[tmp_model_name].append(reload_model(model_name=tmp_model_name, model_version=tmp_version))
        tmp_model = reload_model(model_name=model_name, model_version=tmp_version)
        result.append(tmp_model.predict(test_data))
    with open(result_file,'wb') as f:
        pickle.dump(result, f)

if __name__ == '__main__':
    main()