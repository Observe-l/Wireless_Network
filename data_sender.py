import pandas as pd
import numpy as np
import time
from pathlib import Path
from utils.udp_req import udp_send, udp_server
import optparse

<<<<<<< HEAD
TRAIN_DATA = 'CMAPSSData/train_FD001.txt'

def get_options():
    optParse = optparse.OptionParser()
    optParse.add_option("-p","--port",default="10650",type=int,help="sent to port")
=======
# TRAIN_DATA = 'train_data.txt'
TRAIN_DATA = 'CMAPSSData/train_FD004.txt'

def get_options():
    optParse = optparse.OptionParser()
    optParse.add_option("-p","--port",default="4700",type=int,help="sent to port")
>>>>>>> Backup the yellow computer
    options, args = optParse.parse_args()
    return options

def start_sending():
    options = get_options()
    send_data = open(TRAIN_DATA,'rb')
    data_msg = send_data.readlines()
    buf_msg = b'36 test test \n'
    i = 1
    print(f'Machine {i} start')
    for tmp_msg in data_msg:
        # Only send one unit data. After that, wait 90s
        if tmp_msg.decode('utf-8').split(' ',1)[0] != buf_msg.decode('utf-8').split(' ',1)[0]:
            print(f'Machine {i} stop')
            i += 1
<<<<<<< HEAD
            time.sleep(90)
=======
            # time.sleep(400)
>>>>>>> Backup the yellow computer
            print(f'Machine {i} start')

        buf_msg = tmp_msg
        udp_send(tmp_msg, '172.26.47.188',options.port)
        time.sleep(0.005)

if __name__ == '__main__':
<<<<<<< HEAD
    start_sending()
=======
    start_sending()
>>>>>>> Backup the yellow computer
