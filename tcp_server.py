import sys
import socket

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

sock.bind(("localhost",4700))

sock.listen(1)

file_name = 'receive.txt'

with open(file_name,'wb') as f:
    c, addr = sock.accept()
    while True:
        byte_read = c.recv(4096)
        if not byte_read:
            break
        f.write(byte_read)

sock.close()