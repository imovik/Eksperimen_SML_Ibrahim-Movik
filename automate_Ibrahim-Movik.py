import pandas as pd
import os


def load_and_preprocess_data(raw_data_path):
    """
    Fungsi untuk memuat data mentah dan melakukan preprocessing otomatis.
    Mengembalikan data fitur (X) dan target (y).
    """
    print("Memulai otomatisasi preprocessing data...")

    # 1. Data Loading
    print("1. Memuat dataset mentah...")
    # Menggunakan decimal=',' karena format angka eropa pada dataset
    df = pd.read_csv(raw_data_path, decimal=",", parse_dates=["date"])

    # 2. Data Preprocessing
    print("2. Melakukan data preprocessing...")

    # Menghapus kolom 'date' (tidak dibutuhkan untuk regresi standar)
    if "date" in df.columns:
        df = df.drop("date", axis=1)

    # Menghapus kolom '% Iron Concentrate' (mencegah data leakage)
    if "% Iron Concentrate" in df.columns:
        df = df.drop("% Iron Concentrate", axis=1)

    # Menghapus duplikat
    df = df.drop_duplicates()

    # 3. Splitting X dan y
    print("3. Memisahkan fitur (X) dan target (y)...")
    X = df.drop("% Silica Concentrate", axis=1)
    y = df["% Silica Concentrate"]

    print("Otomatisasi preprocessing selesai!")
    return X, y


if __name__ == "__main__":
    # Menentukan path file mentah (sesuaikan dengan nama file di dalam data_raw)
    file_path = "data_raw/MiningProcess.zip"

    # Mengeksekusi fungsi
    if os.path.exists(file_path):
        X_train, y_train = load_and_preprocess_data(file_path)
        print(f"Bentuk data X_train yang siap dilatih: {X_train.shape}")
        print(f"Bentuk data y_train yang siap dilatih: {y_train.shape}")
    else:
        print(f"Error: File tidak ditemukan di path {file_path}")
