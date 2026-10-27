# MakineNabız AI

Makine çalışma değerlerinden arıza riskini tahmin eden Streamlit uygulaması (KNN, UCI AI4I 2020 veri kümesi).

- `app.py` — arayüz (Arıza Analizi, Senaryo Karşılaştırma)
- `src/` — girdi doğrulama, model yükleme, senaryo karşılaştırma
- `models/` — eğitilmiş model (`best_model.joblib`) ve bilgileri
- `requirements.txt` — sabit paket sürümleri (Python 3.12)

Yerelde çalıştırma: `pip install -r requirements.txt` ve `streamlit run app.py`
