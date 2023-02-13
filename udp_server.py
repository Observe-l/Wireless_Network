import socket
import select

UDP_IP = ""
IN_PORT = 4700
timeout = 3


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, IN_PORT))
file_name = 'receive.txt'

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