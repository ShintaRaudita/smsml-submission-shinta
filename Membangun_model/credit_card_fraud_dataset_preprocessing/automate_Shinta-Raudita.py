import os
import pandas as pd

def automated_preprocessing(input_path, output_path):
    print("=== MULAI OTOMATISASI DATA PREPROCESSING ===")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset mentah tidak ditemukan di: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Dataset berhasil dimuat. Ukuran awal: {df.shape}")
    
    # 2. Menangani/menghapus kolom identifier yang tidak punya nilai prediktif
    df_clean = df.drop(columns=['TransactionID', 'MerchantID'], errors='ignore')
    
    # 3. Rekayasa fitur waktu (feature engineering dari TransactionDate)
    df_clean['TransactionDate'] = pd.to_datetime(df_clean['TransactionDate'])
    df_clean['Hour'] = df_clean['TransactionDate'].dt.hour
    df_clean['DayOfWeek'] = df_clean['TransactionDate'].dt.dayofweek
    
    # Hapus kolom aslinya karena sudah diwakili oleh fitur Hour dan DayOfWeek
    df_clean = df_clean.drop(columns=['TransactionDate'], errors='ignore')
    
    # 4. Encoding data kategorikal (TransactionType & Location) sesuai notebook eksperimen
    df_clean = pd.get_dummies(df_clean, columns=['TransactionType', 'Location'], drop_first=True)
    
    # 5. Memastikan folder direktori tujuan tersedia
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 6. Menyimpan hasil akhir data bersih untuk retraining
    df_clean.to_csv(output_path, index=False)
    print(f"Proses otomatisasi sukses! Ukuran akhir data: {df_clean.shape}")
    print(f"File bersih berhasil disimpan di: {output_path}")

if __name__ == "__main__":
    INPUT_FILE = "namadataset_raw/credit_card_fraud_dataset.csv"
    OUTPUT_FILE = "namadataset_preprocessing/credit_card_fraud_clean.csv"
    
    automated_preprocessing(INPUT_FILE, OUTPUT_FILE)