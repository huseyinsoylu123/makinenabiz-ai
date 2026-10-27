"""Veri yükleme, doğrulama ve bölme işlemleri (AI4I 2020)."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"

CATEGORICAL_FEATURES = ["Type"]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "Machine failure"
# Model girdisi olarak ASLA kullanılmayacak sütunlar (veri sızıntısı / kimlik)
LEAKAGE_COLUMNS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF", TARGET]
EXPECTED_COLUMNS = ["UDI", "Product ID"] + FEATURES + [TARGET, "TWF", "HDF", "PWF", "OSF", "RNF"]

VALID_TYPES = ("L", "M", "H")
# Kabul edilen fiziksel aralıklar: veri kümesindeki min/maks değerlerin biraz dışını kapsar.
# Aralık dışı değerlerde model ekstrapolasyon yapacağından analiz reddedilir.
VALID_RANGES = {
    "Air temperature [K]": (290.0, 310.0),
    "Process temperature [K]": (300.0, 320.0),
    "Rotational speed [rpm]": (1000.0, 3000.0),
    "Torque [Nm]": (0.0, 90.0),
    "Tool wear [min]": (0.0, 260.0),
}
RANDOM_STATE = 42


def load_data(path=DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Beklenen sütunlar eksik: {missing}")
    return df


def get_X_y(df: pd.DataFrame):
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    return X, y


def split_data(X, y, random_state=RANDOM_STATE):
    """Stratified %60 eğitim / %20 doğrulama / %20 test."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


def validate_input(values: dict) -> pd.DataFrame:
    """Kullanıcı girdisini doğrular, tek satırlık DataFrame döndürür. Hatada ValueError."""
    errors = []
    t = values.get("Type")
    if t not in VALID_TYPES:
        errors.append(f"Ürün kalite türü L, M veya H olmalıdır (girilen: {t!r}).")
    row = {"Type": t}
    for col in NUMERIC_FEATURES:
        v = values.get(col)
        try:
            v = float(v)
        except (TypeError, ValueError):
            errors.append(f"{col} sayısal olmalıdır (girilen: {v!r}).")
            continue
        if v != v:  # NaN
            errors.append(f"{col} boş bırakılamaz.")
            continue
        lo, hi = VALID_RANGES[col]
        if not lo <= v <= hi:
            errors.append(f"{col} {lo:g} ile {hi:g} arasında olmalıdır (girilen: {v:g}).")
        row[col] = v
    if not errors and row["Process temperature [K]"] <= row["Air temperature [K]"]:
        errors.append("İşlem sıcaklığı hava sıcaklığından yüksek olmalıdır (veri kümesindeki tüm kayıtlarda böyledir).")
    if errors:
        raise ValueError(" ".join(errors))
    return pd.DataFrame([row], columns=FEATURES)


def kelvin_to_celsius(k: float) -> float:
    return k - 273.15


def celsius_to_kelvin(c: float) -> float:
    return c + 273.15
