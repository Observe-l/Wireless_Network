# Wireless_Network
OpenWRT, Network Control, Wireless Dissipation

#### Start ML Flow Server

```shell
sudo mount -t nfs localhost:/nfsroot ~/nfsroot -o nolock,soft,timeo=30,retry=3

mlflow server \
    --backend-store-uri mysql://lwh:666888@localhost:3306/MLflow \
    --default-artifact-root /home/lwh/nfsroot/mlruns \
    --host 0.0.0.0 \
    --port 5000
    
```

