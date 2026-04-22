import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    Quản lý và tiền xử lý nhiều dataset (all, batbuoc, tuchon)
    Bao gồm: xử lý NaN, encode label, và train/test split.
    """

    def __init__(self):
        """
        Khởi tạo và load dữ liệu từ các file CSV vào dictionary datasets.
        """
        self.datasets = {
            "all": pd.read_csv("dataset/dataset_diem_cntt_all.csv"),
            "batbuoc": pd.read_csv("dataset/dataset_diem_cntt_batbuoc.csv"),
            "tuchon": pd.read_csv("dataset/dataset_diem_cntt_tuchon.csv"),
        }

        self.label_col = "class"

        self.processed_datasets = {}
        self.splitted_datasets = {}

        self.label_encoders = {}

    def _fillna(self, df):
        """
        Thay thế các giá trị NaN bằng 0 để biểu diễn việc sinh viên không học môn.
        """
        return df.fillna(0)

    def _split_xy(self, df):
        """
        Tách dataset thành đặc trưng (X) và nhãn (y).
        """
        X = df.drop(columns=[self.label_col])
        y = df[self.label_col]
        return X, y

    def _encode_label(self, y, name):
        """
        Mã hóa nhãn dạng text (TB, K, Y, ...) thành số nguyên.
        Lưu encoder để có thể decode sau này.
        """
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        self.label_encoders[name] = le
        return y_encoded

    def fit(self):
        """
        Thực hiện toàn bộ bước tiền xử lý:
        - Fill NaN
        - Tách X, y
        - Encode label
        """
        for name, df in self.datasets.items():
            print(f"[INFO] Processing dataset: {name}")

            df = self._fillna(df)

            X, y = self._split_xy(df)

            y_encoded = self._encode_label(y, name)

            self.processed_datasets[name] = {"X": X, "y": y_encoded}

        return self

    def split(self, test_size=0.2, random_state=42):
        """Chia dữ liệu mà không stratify"""
        self.splitted_datasets = {}

        for name, data in self.processed_datasets.items():
            X = data["X"]
            y = data["y"]

            # Bỏ tham số stratify
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,  # Không có stratify
            )

            self.splitted_datasets[name] = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            }

        return self

    def fit_split(self, test_size=0.2):
        """
        Thực hiện liên tiếp fit() và split() để trả về dữ liệu sẵn sàng cho training.
        """
        self.fit()
        self.split(test_size=test_size)
        return self.splitted_datasets

    def summary(self):
        """
        In ra thông tin tổng quan của từng dataset: shape và số lượng giá trị thiếu.
        """
        for name, df in self.datasets.items():
            print(f"\nDataset: {name}")
            print(f"Shape: {df.shape}")
            print(f"Missing values: {df.isnull().sum().sum()}")
