#! /bin/bash
while getopts l:f:w: flag
do
    case "${flag}" in
        l) loss=${OPTARG};;
        f) file=${OPTARG};;
        w) window=${OPTARG};;
    esac
done

source ~/.tf.sh
python3 utils/trainer_regression_dnn.py -l $loss -f $file -w $window
python3 utils/trainer_regression_lstm.py -l $loss -f $file -w $window
python3 utils/trainer_regression_bilstm.py -l $loss -f $file -w $window
python3 utils/trainer_regression_cnn.py -l $loss -f $file -w $window
