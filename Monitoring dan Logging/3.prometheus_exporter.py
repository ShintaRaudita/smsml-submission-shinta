import time
import random
from flask import Flask, Response
from prometheus_client import generate_latest, Counter, Gauge, REGISTRY

app = Flask(__name__)

# Mengunci 10 metrik berbeda
REQUEST_COUNT = Counter('http_requests_total', 'Total jumlah request API')
PREDICTION_FRAUD = Counter('fraud_predictions_total', 'Total transaksi terdeteksi Fraud')
PREDICTION_LEGIT = Counter('legitimate_predictions_total', 'Total transaksi terdeteksi Legit')
ERROR_COUNT = Counter('prediction_error_rate', 'Total kegagalan inferensi model')

CPU_USAGE = Gauge('system_cpu_usage', 'Penggunaan CPU dalam persen')
RAM_USAGE = Gauge('system_ram_usage', 'Penggunaan RAM dalam persen')
LATENCY = Gauge('model_prediction_latency_seconds', 'Kecepatan pemrosesan prediksi model')
AVG_AMOUNT = Gauge('average_transaction_amount', 'Rata-rata nominal uang transaksi yang masuk')
HTTP_200 = Counter('http_status_200_total', 'Total response status 200 OK')
HTTP_500 = Counter('http_status_500_total', 'Total response status 500 Error')

@app.route('/metrics')
def metrics():
    # Menghasilkan data simulasi otomatis setiap kali halaman direfresh
    REQUEST_COUNT.inc(random.randint(10, 50))
    PREDICTION_FRAUD.inc(random.randint(1, 3))
    PREDICTION_LEGIT.inc(random.randint(9, 45))
    HTTP_200.inc(random.randint(10, 50))
    
    CPU_USAGE.set(random.uniform(20.0, 65.0))
    RAM_USAGE.set(random.uniform(70.0, 88.0))
    LATENCY.set(random.uniform(0.01, 0.15))
    AVG_AMOUNT.set(random.uniform(150.0, 250.0))
    
    return Response(generate_latest(REGISTRY), mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)