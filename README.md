# Wireless_Network
OpenWRT, Network Control, Wireless Dissipation

#### Start ML Flow Server

```shell
mlflow server \
    --backend-store-uri mysql://lwh:666888@localhost:3306/MLflow \
    --default-artifact-root /nfsroot/mlruns \
    --host 0.0.0.0 \
    --port 5000
```

