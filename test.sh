#! /bin/bash
while getopts l:f: flag
do
    case "${flag}" in
        l) loss=${OPTARG};;
        f) file=${OPTARG};;
    esac
done

source ~/.tf.sh
python3 model_eva.py -m 1 -a 0 -l $loss -f $file
python3 model_eva.py -m 1 -a 2 -l $loss -f $file
python3 model_eva.py -m 1 -a 3 -l $loss -f $file
python3 model_eva.py -m 1 -a 1 -l $loss -f $file
