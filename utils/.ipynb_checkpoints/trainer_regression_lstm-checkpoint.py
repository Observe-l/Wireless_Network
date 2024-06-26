import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow import keras
# from tensorflow.keras import layers
from sklearn import preprocessing
import mlflow
import socket
import json
import pickle

from udp_req import udp_send, udp_server

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

MODEL_PORT = 10652
DATA_PORT = 10653

FILE_NAME = "re_train.txt"
DATA_TMP = "re_rec.txt"
ORIGIN_DATA = '/home/lwh/Documents/Code/Wireless_Network/CMAPSSData/train_FD001.txt'
TEST_DATA = '/home/lwh/Documents/Code/Wireless_Network/CMAPSSData/test_FD001.txt'
RUL_FILE = "/home/lwh/Documents/Code/Wireless_Network/CMAPSSData/RUL_FD001.txt"
TIME_OUT = 3
MODEL_NAME = "regression_test"
REMAIN_NUM = 100

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def single_train(x_train,y_train):
    client = mlflow.tracking.MlflowClient()
    mlflow.set_experiment("Regression Evaluate")

    i_shape = [25, 25]

    # y_train = keras.utils.to_categorical(label, num_classes)
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.tensorflow.autolog(log_models=True, disable=False, registered_model_name=None)
        model = keras.Sequential(
            [
                keras.layers.LSTM(32,return_sequences=True,activation='tanh',input_shape=i_shape),
                keras.layers.LSTM(64,return_sequences=True,activation='tanh'),
                keras.layers.LSTM(32,return_sequences=False,activation='tanh'),
                keras.layers.Dense(100, activation="relu"),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(1,activation='linear')
            ]
        )

        batch_size = 128
        epochs = 20

        model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])

        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.05)
        # mlflow.sklearn.log_model(model,"model")
        # score = model.evaluate(x_test, y_test, verbose=0)
        # score = -history.history['accuracy'][-1]
    result = mlflow.register_model(
        f"runs:/{run_id}/model",
        MODEL_NAME
    )
    # Get the latest model version
    latest_version = int(result.version)
    client.update_model_version(
        name=MODEL_NAME,
        version=latest_version,
        description='None'

    )
    # Transition the latest model to Production stage, others to Archived stage
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version= latest_version,
        stage='Production',
        archive_existing_versions=True
    )
    return latest_version


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

def update_data(train_data:pd.DataFrame):
    # get latest machine id
    tmp_idx = np.sort(train_data[0].unique())[::-1][:REMAIN_NUM]
    # update file
    train_data.loc[train_data[0].isin(tmp_idx)].to_csv(FILE_NAME, sep=' ', header=False,index=False)

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    start_flag = True
    # initialize data file
    f = open(DATA_TMP,'w')
    f.close()
    # Init data file
    tmp_data = pd.read_csv(ORIGIN_DATA, sep=" ",header=None)
    tmp_idx = np.sort(tmp_data[0].unique())[:REMAIN_NUM]
    tmp_data.loc[tmp_data[0].isin(tmp_idx)].to_csv(FILE_NAME, sep=' ', header=False,index=False)
    tmp_data = pd.read_csv(FILE_NAME, sep=" ",header=None)
    # Init model
    x_train, y_train = data_load()
    model_version = single_train(x_train, y_train)
    sock.connect(("localhost",MODEL_PORT))


    while(True):
        data, addr = udp_server(port=DATA_PORT,timeout=TIME_OUT,start_flag=start_flag)
        if data.decode('utf-8') == 'complete':
            tmp_buf = pd.read_csv(DATA_TMP, sep=" ",header=None)
            tmp_data = pd.concat([tmp_data, tmp_buf])
            # initialize data file
            f = open(DATA_TMP,'w')
            f.close()
            # update train data
            update_data(tmp_data)
            start_flag = True
            x_train, y_train = data_load()
            print(x_train.shape)
            # model_version = hyper_opt(x_train=x_train,y_train=y_train,maxevals=2)
            model_version = single_train(x_train, y_train)
            tmp_train_msg = f'model_version: {model_version}'
            sock.sendall(tmp_train_msg.encode())
            print(f"Retraining completed. The latest model version is: {model_version}")
        else:
            start_flag = False
            with open(DATA_TMP,'a') as f:
                f.write(data.decode('utf-8'))

def model_train():
    x_train, y_train = data_load(file_name=ORIGIN_DATA)
    model_version = single_train(x_train, y_train)
    print(f"Training completed. The latest model version is: {model_version}")

def model_eva():
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{MODEL_NAME}/Production"
    )
    # testdata, testlabel = data_load(file_name=TEST_DATA,model_test=True)
    testdata, testlabel = data_load(file_name=ORIGIN_DATA)

    label_pre = model.predict(testdata)[:,0]

    result_rmse = root_mean_squared_error(testlabel,label_pre)
    pre_rmse = 'reult is:' + str(result_rmse.numpy()) + '\n'
    print(pre_rmse)

mlflow.set_tracking_uri("http://localhost:5000")

model_eva()

if __name__ == "__main__":
    # Tracking the mysql database
    mlflow.set_tracking_uri("http://localhost:5000")
    # main()
    # model_train()
    model_eva()

