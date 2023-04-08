#! /bin/bash
while getopts l:f: flag
do
    case "${flag}" in
        l) loss=${OPTARG};;
        f) file=${OPTARG};;
    esac
done

source ~/.tf.sh
python3 utils/trainer_regression_dnn.py -l $loss -f $file
python3 utils/trainer_regression_lstm.py -l $loss -f $file
python3 utils/trainer_regression_bilstm.py -l $loss -f $file
python3 utils/trainer_regression_cnn.py -l $loss -f $file
