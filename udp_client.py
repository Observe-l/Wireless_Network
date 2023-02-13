import socket
import os
import tqdm
import time

# UDP_IP = "192.168.1.133"
UDP_IP = "172.26.49.0"
UDP_PORT = 4700
buf = 1024
file_name = 'CMAPSSData/train_FD003.txt'
file_size = os.path.getsize(file_name)
progress = tqdm.tqdm(range(file_size), f"Sending {file_name}", unit="B", unit_scale=True, unit_divisor=1024)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

with open(file_name, "rb") as f:
    while True:
        data = f.read(buf)
        if not data:
            break
        if(sock.sendto(data, (UDP_IP, UDP_PORT))):
            progress.update(len(data))
            time.sleep(0.002)

sock.close()