import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import dagshub

def train_base_model():
    # Inisialisasi online tracking ke dagsub
    try:
        dagshub.init(repo_owner='ShintaRaudita', repo_name='mlsystem-credit-card-fraud', mlflow=True)
    except Exception as e:
        print(f"[!] DagsHub init gagal: {e}. Menggunakan tracking lokal.")
        mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    
    mlflow.set_experiment("Credit Card Fraud Detection")
    
    # Load data hasil preprocessing
    data_path = "credit_card_fraud_dataset_preprocessing/credit_card_fraud_clean.csv"
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['IsFraud'])
    y = df['IsFraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Model baseline
    print("[-] Melatih Baseline Model...")
    model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Hitung metrik evaluasi
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Confusion Matrix
    plt.figure(figsize=(5,4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.title('Baseline Confusion Matrix')
    cm_path = "training_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    
    metric_info_path = "metric_info.json"
    with open(metric_info_path, "w") as f:
        f.write(f'{{"precision": {prec}, "recall": {rec}, "f1_score": {f1}}}')

    # Logging manual ke mlflow
    with mlflow.start_run(run_name="Baseline_Model"):
        mlflow.log_param("n_estimators", 20)
        mlflow.log_param("max_depth", 5)
        
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # Simpan Artefak Tambahan ke MLflow UI
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(metric_info_path)
        
        # Simpan model mlflow ke folder 'model'
        mlflow.sklearn.log_model(model, "model")
        print("Baseline model berhasil dicatat di MLflow")

if __name__ == "__main__":
    train_base_model()
    
    
    
    
    
    
    
    
    