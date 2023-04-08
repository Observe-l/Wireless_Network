import socket
import select

UDP_IP = ""
IN_PORT = 4700
timeout = 3
import optparse

def get_options():
    optParse = optparse.OptionParser()
    optParse.add_option("-l","--loss",default="0",type=str,help="loss_rate")
    optParse.add_option("-f","--file",default="FD001",type=str,help="trainning dataset")
    options, args = optParse.parse_args()
    return options

options = get_options()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, IN_PORT))
file_name = 'train_data/' + options.file + '/loss' + options.loss + '_train.txt'

with open(file_name, 'wb') as f:
    data, addr = sock.recvfrom(1024)
    f.write(data)
    while True:
        ready = select.select([sock], [], [], timeout)
        if ready[0]:
            data, addr = sock.recvfrom(1024)
            f.write(data)
        else:
            print("Finish!")
            break

sock.close()