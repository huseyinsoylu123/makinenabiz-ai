"""Sensör senaryo simülatörü: tek bir sensörü değiştirip modeli iki kez çalıştırır.

Bu, modelin farklı girdilere yanıtını inceleyen bir karşılaştırmadır; nedensel bir
bakım önerisi değildir.
"""
from src.data_processing import NUMERIC_FEATURES
from src.inference import predict


def compare_scenarios(model, metadata, base_values: dict, sensor: str, new_value: float) -> dict:
    if sensor not in NUMERIC_FEATURES:
        raise ValueError(f"Değiştirilebilir sensör değil: {sensor}")
    scenario_b = dict(base_values)
    scenario_b[sensor] = new_value
    a = predict(model, metadata, base_values)
    b = predict(model, metadata, scenario_b)
    return {"sensor": sensor, "old_value": float(base_values[sensor]), "new_value": float(new_value),
            "scenario_a": a, "scenario_b": b,
            "probability_delta": b["failure_probability"] - a["failure_probability"],
            "class_changed": a["predicted_class"] != b["predicted_class"]}
