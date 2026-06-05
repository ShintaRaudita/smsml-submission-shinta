import time
import random
from fastapi import FastAPI
import requests

app = FastAPI()

@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    # Simulasi penerimaan request data transaksi
    time.sleep(random.uniform(0.01, 0.2)) 
    
    # Kirim sinyal update ke Prometheus Exporter via hit dummy
    latency = time.time() - start_time
    print(f"[Inferensi] Latensi: {latency}s")
    return {"status": "success", "prediction": random.choice([0, 1])}



