"""MakineNabız AI — Streamlit Community Cloud sürümü.  Çalıştırma: streamlit run app.py"""
import altair as alt
import pandas as pd
import streamlit as st

from src.data_processing import NUMERIC_FEATURES, VALID_RANGES, kelvin_to_celsius
from src.inference import ModelNotAvailableError, load_evaluation, load_model, predict
from src.simulator import compare_scenarios

st.set_page_config(page_title="MakineNabız AI", layout="wide")

TR = {
    "Type": "Ürün kalite türü",
    "Air temperature [K]": "Hava sıcaklığı (K)",
    "Process temperature [K]": "İşlem sıcaklığı (K)",
    "Rotational speed [rpm]": "Dönüş hızı (rpm)",
    "Torque [Nm]": "Tork (Nm)",
    "Tool wear [min]": "Takım aşınması (dk)",
}
DEFAULTS = {"Type": "L", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500.0, "Torque [Nm]": 40.0, "Tool wear [min]": 120.0}
STEPS = {"Air temperature [K]": 0.1, "Process temperature [K]": 0.1, "Rotational speed [rpm]": 10.0,
         "Torque [Nm]": 0.5, "Tool wear [min]": 1.0}

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1100px;}
.mn-header {background: linear-gradient(90deg,#13233b,#0e1726); border:1px solid #24364f;
  border-left: 5px solid #22c3c3; border-radius: 10px; padding: 18px 22px; margin-bottom: 14px;}
.mn-header h1 {margin:0; font-size: 2rem; color:#e6edf3;}
.mn-header p {margin:4px 0 0; color:#9fb3c8;}
.mn-card {background:#1a2433; border:1px solid #2a3a50; border-radius:10px; padding:14px 16px; height:100%;}
.mn-card .lbl {color:#9fb3c8; font-size:.8rem; text-transform:uppercase; letter-spacing:.05em;}
.mn-card .val {font-size:1.6rem; font-weight:700; margin-top:2px;}
.ok {color:#3ddc97;} .warn {color:#f2b84b;} .muted {color:#9fb3c8;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="mn-header"><h1>MakineNabız AI</h1>'
            '<p>Endüstriyel Makine Arıza Analizi ve Senaryo Simülatörü</p></div>', unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return load_model()


try:
    MODEL, META = get_model()
except ModelNotAvailableError as exc:
    st.error(f"Model yüklenemedi, tahmin yapılamaz. {exc}")
    st.stop()
try:
    DEMOS = load_evaluation().get("demo_examples", [])
except (OSError, ValueError):
    DEMOS = []
THR = META["decision_threshold"]


def card(label, value, cls=""):
    st.markdown(f'<div class="mn-card"><div class="lbl">{label}</div>'
                f'<div class="val {cls}">{value}</div></div>', unsafe_allow_html=True)


def sensor_form(prefix, defaults):
    vals = {}
    c1, c2, c3 = st.columns(3)
    with c1:
        vals["Type"] = st.selectbox(TR["Type"], ["L", "M", "H"], index=["L", "M", "H"].index(defaults["Type"]),
                                    key=f"{prefix}_type", help="L: düşük, M: orta, H: yüksek kalite ürün")
    for col, feat in zip([c2, c3, c1, c2, c3], NUMERIC_FEATURES):
        lo, hi = VALID_RANGES[feat]
        with col:
            vals[feat] = st.number_input(f"{TR[feat]}  [{lo:g}–{hi:g}]", value=float(defaults[feat]),
                                         step=STEPS[feat], key=f"{prefix}_{feat}", format="%.1f")
    return vals


tab1, tab2 = st.tabs(["Arıza Analizi", "Senaryo Karşılaştırma"])

with tab1:
    st.subheader("Makine Arıza Analizi")
    demo_opts = {"Varsayılan değerler": DEFAULTS}
    for d in DEMOS:
        name = "Örnek: riskli makine" if "yüksek" in d["etiket"] else "Örnek: sorunsuz makine"
        demo_opts[f"{name} (kayıt {d['UDI']})"] = d["girdi"]
    demo = st.radio("Hangi değerlerle başlayalım?", list(demo_opts), horizontal=True)
    demo_key = list(demo_opts).index(demo)
    with st.form("analysis"):
        values = sensor_form(f"a{demo_key}", demo_opts[demo])
        submitted = st.form_submit_button("Arıza Analizini Başlat", type="primary", use_container_width=True)
    if submitted:
        try:
            r = predict(MODEL, META, values)
        except ValueError as exc:
            st.error(f"Geçersiz giriş: {exc}")
        else:
            p = r["failure_probability"]
            is_fail = r["predicted_class"] == 1
            c1, c2, c3 = st.columns(3)
            with c1:
                card("Sonuç", "ARIZA RİSKİ VAR" if is_fail else "NORMAL", "warn" if is_fail else "ok")
            with c2:
                card("Arıza ihtimali", f"%{p*100:.1f}")
            with c3:
                card("Alarm sınırı", f"%{THR*100:.0f}", "muted")
            st.write("")
            shown = {TR[k]: v for k, v in r["input"].items()}
            shown["Hava sıcaklığı (°C)"] = round(kelvin_to_celsius(r["input"]["Air temperature [K]"]), 2)
            shown["İşlem sıcaklığı (°C)"] = round(kelvin_to_celsius(r["input"]["Process temperature [K]"]), 2)
            st.dataframe(pd.DataFrame([shown]), hide_index=True, use_container_width=True)
            st.info(f"Model, girdiğiniz değerlere en çok benzeyen {META['hyperparameters'].get('n_neighbors')} kayda baktı ve "
                    f"arıza ihtimalini **%{p*100:.1f}** buldu. Alarm sınırı %{THR*100:.0f} olduğu için sonuç "
                    f"**{'arıza riski var' if is_fail else 'normal'}** olarak gösteriliyor.")

with tab2:
    st.subheader("Senaryo Karşılaştırma")
    st.markdown("**Senaryo A: şu anki durum**")
    base = sensor_form("s", {**DEFAULTS, "Rotational speed [rpm]": 1400.0, "Torque [Nm]": 55.0, "Tool wear [min]": 120.0})
    c1, c2 = st.columns(2)
    with c1:
        sensor = st.selectbox("Hangi değeri değiştirelim?", NUMERIC_FEATURES, index=4, format_func=lambda f: TR[f])
    with c2:
        lo, hi = VALID_RANGES[sensor]
        default_new = 220.0 if sensor == "Tool wear [min]" else float(base[sensor])
        new_val = st.number_input(f"Senaryo B: yeni {TR[sensor]} [{lo:g}–{hi:g}]", value=default_new,
                                  step=STEPS[sensor], key=f"new_{sensor}", format="%.1f")
    if st.button("Senaryoları Karşılaştır", type="primary", use_container_width=True):
        try:
            cmp = compare_scenarios(MODEL, META, base, sensor, new_val)
        except ValueError as exc:
            st.error(f"Geçersiz giriş: {exc}")
        else:
            a, b = cmp["scenario_a"], cmp["scenario_b"]
            d = cmp["probability_delta"]
            c1, c2, c3 = st.columns(3)
            with c1:
                card("Senaryo A", f"%{a['failure_probability']*100:.1f} · {'RİSKLİ' if a['predicted_class'] else 'NORMAL'}",
                     "warn" if a["predicted_class"] else "ok")
            with c2:
                card("Senaryo B", f"%{b['failure_probability']*100:.1f} · {'RİSKLİ' if b['predicted_class'] else 'NORMAL'}",
                     "warn" if b["predicted_class"] else "ok")
            with c3:
                card("Değişim", f"{d*100:+.1f} puan", "warn" if d > 0 else "ok")
            chart_df = pd.DataFrame({"Senaryo": ["A (önce)", "B (sonra)"],
                                     "Arıza ihtimali (%)": [a["failure_probability"]*100, b["failure_probability"]*100]})
            bars = alt.Chart(chart_df).mark_bar(size=60, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("Senaryo:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Arıza ihtimali (%):Q", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Senaryo:N", scale=alt.Scale(range=["#22c3c3", "#f2b84b"]), legend=None))
            rule = alt.Chart(pd.DataFrame({"y": [THR*100]})).mark_rule(strokeDash=[6, 4], color="#e66").encode(
                y=alt.Y("y:Q", title=None))
            st.altair_chart((bars + rule).properties(height=320), use_container_width=True)
            st.caption(f"Kesikli kırmızı çizgi: alarm sınırı (%{THR*100:.0f}).")
            if not cmp["class_changed"]:
                st.info("Sonuç değişmedi; bu değişiklik alarm sınırını aşacak kadar etkili olmadı.")
