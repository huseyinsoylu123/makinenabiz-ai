"""Kayıtlı modeli yükleme ve tahmin. Model yoksa sahte tahmin yerine hata verilir."""
import json
from pathlib import Path

import joblib

from src.data_processing import FEATURES, PROJECT_ROOT, validate_input

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"
EVALUATION_PATH = PROJECT_ROOT / "models" / "evaluation.json"


class ModelNotAvailableError(RuntimeError):
    """Model veya metadata dosyası eksik/bozuk."""


def load_model(model_path=MODEL_PATH, metadata_path=METADATA_PATH):
    model_path, metadata_path = Path(model_path), Path(metadata_path)
    if not model_path.exists() or not metadata_path.exists():
        raise ModelNotAvailableError(
            f"Model dosyaları bulunamadı ({model_path.name}, {metadata_path.name}). "
            "Önce 'python -m src.train_model' komutuyla modeli eğitin.")
    try:
        model = joblib.load(model_path)
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as exc:  # bozuk dosya, sürüm uyumsuzluğu vb.
        raise ModelNotAvailableError(f"Model yüklenemedi: {exc}") from exc
    if metadata.get("features") != FEATURES or "decision_threshold" not in metadata:
        raise ModelNotAvailableError("metadata.json beklenen özellik listesini veya karar eşiğini içermiyor.")
    return model, metadata


def load_evaluation(path=EVALUATION_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def predict(model, metadata, values: dict) -> dict:
    """Girdiyi doğrular ve modeli çalıştırır. Geçersiz girdide ValueError."""
    X = validate_input(values)
    proba = float(model.predict_proba(X)[0, 1])
    threshold = float(metadata["decision_threshold"])
    return {"failure_probability": proba, "threshold": threshold,
            "predicted_class": int(proba >= threshold), "input": X.iloc[0].to_dict()}
