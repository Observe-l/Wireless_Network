import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

sock.connect(("localhost",4700))

with open('CMAPSSData/readme.txt','rb') as f:
    while True:
        bytes_read = f.read(4096)
        if not bytes_read:
            break
        sock.sendall(bytes_read)

sock.close()