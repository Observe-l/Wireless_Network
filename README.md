# Wireless_Network
OpenWRT, Network Control, Wireless Dissipation

```
sudo iptables -A INPUT -m statistic -s 172.16.188.140 --mode random --probability 0.1 -j DROP
sudo iptables -A INPUT -m statistic -s 10.248.11.4 --mode random --probability 0.15 -j DROP
sudo iptables -A INPUT -m statistic -s 10.248.37.23 --mode random --probability 0.3 -j DROP


```

```shell
wdiff -s receive.txt CMAPSSData/train_FD003.txt
```

### Network Congestion

Configuration: 2 MB/s, UDP protocol

| background trafic flow (MB/s) | packet loss | UDP sending speed |
| ----------------------------- | ----------- | ----------------- |
| 0                             | 0%          | 2 MB/s            |
| 0.5                           | 0%          | 2 MB/s            |
| 1                             | 0%          | 2 MB/s            |
| 1.5                           | 0%          |                   |
| 2                             | 14%         |                   |
| 2.5                           | 20%         | 1.7 MB/s          |
| 3                             | 45%         |                   |
| 3.5                           |             |                   |
| unconstrained                 |             |                   |

### Flow Control

Configuration: 2 MB/s, UDP protocol, background flow: TCP

| background trafic flow (MB/s) | packet loss | UDP sending speed |
| ----------------------------- | ----------- | ----------------- |
| 0                             | 0%          | 2 MB/s            |
| 0.5                           | 0%          | 2 MB/s            |
| 1                             | 0%          | 2 MB/s            |
| 1.5                           | 0%          |                   |
| 2.0                           | 0%          | 1.2-1.6           |
| 2.5                           | 0%          |                   |
| 3                             | 4%          |                   |
| 3.5                           | 8%          |                   |
| unconstrained                 |             |                   |
