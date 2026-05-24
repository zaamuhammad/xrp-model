import yfinance as yf
import streamlit as st
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import plotly.graph_objects as go
import joblib
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CONFIG_PATH = "config.json"
def load_config(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"window": 44, "features": ["Close"]}

config = load_config(CONFIG_PATH)
WINDOW = config.get("window", 44)

MODEL_PATH  = "model_lstm_xrp.h5"
SCALER_PATH = "scaler_close.save"

# ─── PAGE ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="XRP LSTM Predictor", page_icon="💹", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

*,
html,
body,
[class*="css"] {
    font-family: 'Syne', sans-serif;
}

header[data-testid="stHeader"] {
    display: none !important;
}

.stDeployButton {
    display: none !important;
}

button[title="View app in Streamlit Community Cloud"] {
    display: none !important;
}

.block-container {
    padding-top: 1rem !important;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(96,165,250,0.20), transparent 30%),
        radial-gradient(circle at bottom right, rgba(59,130,246,0.18), transparent 30%),
        linear-gradient(135deg, #f8fbff 0%, #eef4ff 45%, #e0ecff 100%);
    color: #0f172a;
}

h1, h2, h3, h4, h5, h6 {
    color: #0f172a !important;
    font-weight: 800 !important;
}

.sec {
    font-family:'Syne',sans-serif;
    font-weight:800;
    font-size:1.1rem;
    color:#1e3a8a;
    border-left:5px solid #2563eb;
    padding-left:.8rem;
    margin:1.5rem 0 1rem;
}

.card {
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(255,255,255,0.55);
    border-radius: 22px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    backdrop-filter: blur(16px);
    box-shadow:
        0 8px 30px rgba(59,130,246,0.10),
        inset 0 1px 0 rgba(255,255,255,0.5);
    transition: all .25s ease;
    height: 100%;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow:
        0 14px 38px rgba(37,99,235,0.16),
        inset 0 1px 0 rgba(255,255,255,0.6);
}

.card-lbl {
    font-family:'Space Mono',monospace;
    font-size:.64rem;
    color:#475569;
    font-weight:700;
    letter-spacing:2px;
    text-transform:uppercase;
    margin-bottom:.45rem;
}

.card-val {
    font-family:'Syne',sans-serif;
    font-size:1.75rem;
    font-weight:800;
    color:#1d4ed8;
    line-height:1.1;
}

.card-sub {
    font-family:'Space Mono',monospace;
    font-size:.68rem;
    margin-top:.35rem;
    color:#64748b;
}

.pcard {
    background: rgba(255,255,255,0.75);
    border: 1px solid rgba(255,255,255,0.6);
    border-radius: 22px;
    padding: 1.2rem 1rem;
    text-align: center;
    backdrop-filter: blur(14px);
    box-shadow:
        0 8px 28px rgba(59,130,246,0.10),
        inset 0 1px 0 rgba(255,255,255,0.6);
    transition: all .25s ease;
    height: 100%;
}

.pcard:hover {
    transform: translateY(-4px);
    box-shadow:
        0 14px 36px rgba(37,99,235,0.16),
        inset 0 1px 0 rgba(255,255,255,0.65);
}

.pcard-lbl {
    font-family:'Space Mono',monospace;
    font-size:.58rem;
    color:#64748b;
    font-weight:700;
    letter-spacing:2px;
    text-transform:uppercase;
    margin-bottom:.45rem;
}

.pcard-val {
    font-family:'Syne',sans-serif;
    font-size:1.45rem;
    font-weight:800;
    line-height:1.1;
}

.pcard-sub {
    font-family:'Space Mono', monospace;
    font-size:.68rem;
    margin-top:.35rem;
}

.pos { color:#16a34a; }
.neg { color:#dc2626; }
.neu { color:#d97706; }

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, #1d4ed8 0%, #2563eb 50%, #3b82f6 100%);
    border-right: 1px solid rgba(255,255,255,0.15);
}

.sidebar-title {
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 1.1rem;
}

[data-testid="stSidebar"] label {
    color: #ffffff !important;
    font-weight: 700 !important;
}

[data-testid="stSidebar"] * {
    color: white;
}

div[data-baseweb="select"] {
    background: rgba(255,255,255,0.95);
    border-radius: 12px;
}

div[data-baseweb="select"] span {
    color: #0f172a !important;
}

div[role="option"] {
    color: #0f172a !important;
}

[data-testid="stSelectbox"] * {
    color: #0f172a !important;
}

.stSlider {
    padding-top: .5rem;
}

.stButton>button {
    background: linear-gradient(135deg,#2563eb,#3b82f6);
    color: white;
    border: none;
    border-radius: 14px;
    font-family:'Syne',sans-serif;
    font-weight:700;
    padding:.78rem 1.4rem;
    width:100%;
    transition: all .25s ease;
    box-shadow: 0 6px 18px rgba(37,99,235,0.22);
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(37,99,235,0.30);
}

.stDataFrame {
    background: rgba(255,255,255,0.75);
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.6);
    overflow: hidden;
    backdrop-filter: blur(14px);
    box-shadow:
        0 8px 28px rgba(59,130,246,0.08),
        inset 0 1px 0 rgba(255,255,255,0.6);
}

thead tr th {
    background: linear-gradient(135deg,#2563eb,#3b82f6) !important;
    color: #ffffff !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    border: none !important;
}

tbody tr td {
    background: rgba(255,255,255,0.82) !important;
    color: #0f172a !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.74rem !important;
    border-bottom: 1px solid rgba(148,163,184,0.12) !important;
}

tbody tr:hover td {
    background: rgba(219,234,254,0.72) !important;
}

div[data-testid="stDataFrame"],
div[data-testid="stDataFrame"] > div,
div[data-testid="stDataFrame"] table,
div[data-testid="stDataFrame"] thead,
div[data-testid="stDataFrame"] tbody,
div[data-testid="stDataFrame"] tr,
div[data-testid="stDataFrame"] th,
div[data-testid="stDataFrame"] td {
    color: #0f172a !important;
}

div[data-testid="stDataFrame"] thead th {
    background: linear-gradient(135deg,#2563eb,#3b82f6) !important;
    color: white !important;
}

div[data-testid="stDataFrame"] tbody tr:hover {
    background-color: rgba(219,234,254,0.75) !important;
}

.js-plotly-plot {
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 8px 30px rgba(59,130,246,0.08);
}

.footer {
    text-align:center;
    font-family:'Space Mono',monospace;
    font-size:.62rem;
    color:#64748b;
    margin-top:2rem;
    padding:1rem;
}

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #dbeafe; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(#60a5fa,#2563eb);
    border-radius: 999px;
}
::-webkit-scrollbar-thumb:hover { background: linear-gradient(#3b82f6,#1d4ed8); }
</style>
""", unsafe_allow_html=True)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title"> DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown("---")

    period_map = {
        "1 Bulan":  "1mo",
        "2 Bulan":  "2mo",
        "3 Bulan":  "3mo",
        "6 Bulan":  "6mo",
        "1 Tahun":  "1y",
    }

    period_label = st.selectbox(
        "Periode Historis",
        list(period_map.keys()),
        index=1
    )

    period   = period_map[period_label]
    n_future = st.slider("Prediksi ke depan (hari)", 1, 30, 7)

    st.markdown("---")
    run_btn = st.button("🚀 Jalankan Prediksi")
    st.markdown("---")

# ─── FUNCTIONS ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_data(period):
    import time

    days_map = {"1mo": 30, "2mo": 60, "3mo": 90, "6mo": 180, "1y": 365}
    days = days_map.get(period, 60)

    # Coba endpoint market_chart dulu
    urls = [
        {
            "url": "https://api.coingecko.com/api/v3/coins/ripple/market_chart",
            "params": {"vs_currency": "usd", "days": days, "interval": "daily"},
            "type": "market_chart"
        },
        {
            "url": "https://api.coingecko.com/api/v3/coins/ripple/ohlc",
            "params": {"vs_currency": "usd", "days": days},
            "type": "ohlc"
        }
    ]

    for endpoint in urls:
        for attempt in range(3):
            try:
                time.sleep(2)  # jeda sebelum request
                r = requests.get(endpoint["url"], params=endpoint["params"], timeout=20)
                if r.status_code == 429:
                    time.sleep(15 * (attempt + 1))
                    continue
                if r.status_code != 200:
                    continue
                data = r.json()

                if endpoint["type"] == "ohlc":
                    if isinstance(data, list) and len(data) > 0:
                        df = pd.DataFrame(data, columns=["timestamp", "Open", "High", "Low", "Close"])
                        df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
                        df.set_index("Date", inplace=True)
                        df["Volume"] = 0
                        df = df[["Open", "High", "Low", "Close", "Volume"]]
                        df.dropna(inplace=True)
                        return df

                elif endpoint["type"] == "market_chart":
                    if "prices" in data and len(data["prices"]) > 0:
                        df = pd.DataFrame(data["prices"], columns=["timestamp", "Close"])
                        df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
                        df.set_index("Date", inplace=True)
                        df["Open"]   = df["Close"].shift(1).fillna(df["Close"])
                        df["High"]   = df[["Open", "Close"]].max(axis=1) * 1.01
                        df["Low"]    = df[["Open", "Close"]].min(axis=1) * 0.99
                        df["Volume"] = 0
                        df = df[["Open", "High", "Low", "Close", "Volume"]]
                        df.dropna(inplace=True)
                        return df
            except Exception:
                time.sleep(5)

    raise ValueError("Gagal mengambil data. Coba refresh halaman dalam beberapa menit.")

@st.cache_resource
def load_model_scaler():
    errors = []
    model  = None
    scaler = None

    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        else:
            errors.append(f"❌ Model tidak ditemukan: {MODEL_PATH}")
    except Exception as e:
        errors.append(f"❌ Model error: {str(e)}")

    try:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        else:
            errors.append(f"❌ Scaler tidak ditemukan: {SCALER_PATH}")
    except Exception as e:
        errors.append(f"❌ Scaler error: {str(e)}")

    if model is None or scaler is None:
        errors.append("❌ Model atau scaler gagal load (None)")

    return model, scaler, errors

def predict_future(model, scaler, close_arr, window, n):
    scaled = scaler.transform(close_arr.reshape(-1, 1))
    seq    = scaled[-window:].copy()
    preds  = []
    for _ in range(n):
        x = seq.reshape(1, window, 1)
        p = model.predict(x, verbose=0)[0][0]
        preds.append(p)
        seq = np.vstack([seq[1:], [[p]]])
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

def predict_history(model, scaler, close_arr, window):
    scaled = scaler.transform(close_arr.reshape(-1, 1))
    preds  = []
    for i in range(window, len(scaled)):
        x = scaled[i-window:i].reshape(1, window, 1)
        preds.append(model.predict(x, verbose=0)[0][0])
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

def calc_risk(close_arr):
    ret   = pd.Series(close_arr).pct_change().dropna()
    volat = ret.std() * 100
    if volat > 4:   return "TINGGI 🔴", "neg", "⚠️ Volatilitas tinggi"
    elif volat > 2: return "SEDANG 🟡", "neu", "⚡ Waspadai pergerakan"
    else:           return "RENDAH 🟢", "pos", "✅ Kondisi relatif stabil"

def calc_confidence(close_arr, conn_y):
    MAPE = 3.0846 / 100
    ub, lb = [], []
    for v in conn_y:
        ub.append(v * (1 + MAPE))
        lb.append(v * (1 - MAPE))
    return ub, lb

def line_chart(df_hist, hist_dates, hist_pred, future_dates, future_pred, close_arr):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_hist.index, y=df_hist["Close"],
        name="Harga Aktual",
        line=dict(color="#94a3b8", width=1.8),
        hovertemplate="<b>Aktual</b><br>%{x|%d %b %Y}<br>$%{y:.4f}<extra></extra>"
    ))

    if hist_dates is not None:
        fig.add_trace(go.Scatter(
            x=hist_dates, y=hist_pred,
            name="Prediksi Historis",
            line=dict(color="#f59e0b", width=1.5, dash="dot"),
            hovertemplate="<b>Pred. Historis</b><br>%{x|%d %b %Y}<br>$%{y:.4f}<extra></extra>"
        ))

    if future_pred is not None and len(future_pred):
        conn_x = [df_hist.index[-1]] + future_dates
        conn_y = [df_hist["Close"].iloc[-1]] + list(future_pred)

        ub, lb = calc_confidence(close_arr, conn_y)
        fig.add_trace(go.Scatter(
            x=conn_x + conn_x[::-1],
            y=ub + lb[::-1],
            fill="toself",
            fillcolor="rgba(56,189,248,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Confidence Interval",
            showlegend=True,
            hoverinfo="skip"
        ))

        fig.add_trace(go.Scatter(
            x=conn_x, y=conn_y,
            name="Prediksi Masa Depan",
            line=dict(color="#38bdf8", width=2.5),
            mode="lines+markers",
            marker=dict(size=6, color="#38bdf8"),
            hovertemplate="<b>Prediksi</b><br>%{x|%d %b %Y}<br>$%{y:.4f}<extra></extra>"
        ))

        fig.add_vline(
            x=df_hist.index[-1],
            line_dash="dash", line_color="rgba(255,255,255,0.15)", line_width=1
        )
        fig.add_annotation(
            x=df_hist.index[-1], y=max(conn_y) * 1.01,
            text="▶ Prediksi", showarrow=False,
            font=dict(color="#38bdf8", size=10, family="Space Mono")
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(7,9,15,0)",
        plot_bgcolor="rgba(13,17,27,0.9)",
        font=dict(family="Space Mono", size=11, color="#64748b"),
        legend=dict(bgcolor="rgba(13,17,27,0.8)", bordercolor="#1e3a5f",
                    borderwidth=1, font=dict(size=10)),
        hovermode="x unified",
        height=460,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(gridcolor="#0f1f35", showgrid=True),
        yaxis=dict(gridcolor="#0f1f35", showgrid=True, tickprefix="$", tickformat=".4f"),
    )
    return fig

# ─── FETCH DATA ───────────────────────────────────────────────────────────────
with st.spinner("🔄 Mengambil data XRP/USDT dari Yahoo Finance..."):
    try:
        df = fetch_data(period)
        if len(df) < WINDOW + 2:
            st.error(f"Data terlalu sedikit ({len(df)} baris). Pilih periode lebih panjang.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Gagal fetch data: {e}")
        st.stop()

# ─── MARKET OVERVIEW CARDS ────────────────────────────────────────────────────
cur  = df["Close"].iloc[-1]
prev = df["Close"].iloc[-2]
chg  = cur - prev
chgp = (chg / prev) * 100
h30  = df["High"].tail(30).max()
l30  = df["Low"].tail(30).min()

sg = "▲" if chg >= 0 else "▼"
dc = "pos" if chg >= 0 else "neg"

st.markdown('<div class="sec">Market Overview</div>', unsafe_allow_html=True)

close_arr_tmp = df["Close"].values
risk_label, risk_cls, risk_desc = calc_risk(close_arr_tmp)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""<div class="card">
        <div class="card-lbl">HARGA XRP/USDT</div>
        <div class="card-val">${cur:.4f}</div>
        <div class="card-sub {dc}">{sg} ${abs(chg):.4f} ({abs(chgp):.2f}%) hari ini</div>
    </div>""", unsafe_allow_html=True)

with m2:
    st.markdown(f"""<div class="card">
        <div class="card-lbl">TERTINGGI 30 HARI</div>
        <div class="card-val" style="color:#34d399">${h30:.4f}</div>
        <div class="card-sub pos">▲ +{((h30 - cur) / cur * 100):.2f}% dari skrg</div>
    </div>""", unsafe_allow_html=True)

with m3:
    st.markdown(f"""<div class="card">
        <div class="card-lbl">TERENDAH 30 HARI</div>
        <div class="card-val" style="color:#f87171">${l30:.4f}</div>
        <div class="card-sub neg">▼ {((l30 - cur) / cur * 100):.2f}% dari skrg</div>
    </div>""", unsafe_allow_html=True)

with m4:
    st.markdown(f"""<div class="card">
        <div class="card-lbl">LEVEL RISIKO</div>
        <div class="card-val {risk_cls}" style="font-size:1.35rem">{risk_label}</div>
        <div class="card-sub">{risk_desc}</div>
    </div>""", unsafe_allow_html=True)

# ─── PREDICTION CARDS ─────────────────────────────────────────────────────────
st.markdown('<div class="sec">Analisis Prediksi LSTM</div>', unsafe_allow_html=True)

close_arr = df["Close"].values
risk_label, risk_cls, risk_desc = calc_risk(close_arr)

# Inisialisasi session_state
if "hist_dates"        not in st.session_state: st.session_state.hist_dates        = None
if "hist_pred_arr"     not in st.session_state: st.session_state.hist_pred_arr     = None
if "future_dates"      not in st.session_state: st.session_state.future_dates      = []
if "future_pred"       not in st.session_state: st.session_state.future_pred       = None
if "nxt"               not in st.session_state: st.session_state.nxt               = None
if "pred_high"         not in st.session_state: st.session_state.pred_high         = None
if "pred_low"          not in st.session_state: st.session_state.pred_low          = None

if run_btn:
    model, scaler, errors = load_model_scaler()

    if model is None or scaler is None:
        for e in errors:
            st.error(e)
        st.error("❌ Model/scaler belum berhasil dimuat!")
        st.stop()
    elif errors:
        for e in errors:
            st.error(e)
        st.info("💡 Pastikan file model & scaler ada satu folder dengan app.py")
    elif len(df) < WINDOW:
        st.error(f"Data kurang. Butuh ≥{WINDOW} baris.")
    else:
        with st.spinner(f"Memprediksi {n_future} hari ke depan..."):
            try:
                future_pred   = predict_future(model, scaler, close_arr, WINDOW, n_future)
                future_dates  = [df.index[-1] + timedelta(days=i+1) for i in range(n_future)]
                hist_pred_arr = predict_history(model, scaler, close_arr, WINDOW)
                hist_dates    = df.index[WINDOW:]

                st.session_state.future_pred        = future_pred
                st.session_state.future_dates       = future_dates
                st.session_state.hist_pred_arr      = hist_pred_arr
                st.session_state.hist_dates         = hist_dates
                st.session_state.nxt                = future_pred[0]
                st.session_state.pred_high          = float(max(future_pred))
                st.session_state.pred_low           = float(min(future_pred))
            except Exception as ex:
                st.error(f"❌ Error prediksi: {ex}")
                st.exception(ex)

# Ambil dari session_state
hist_dates    = st.session_state.hist_dates
hist_pred_arr = st.session_state.hist_pred_arr
future_dates  = st.session_state.future_dates
future_pred   = st.session_state.future_pred
nxt           = st.session_state.nxt
pred_high     = st.session_state.pred_high
pred_low      = st.session_state.pred_low

# ─── RISK BANNER ──────────────────────────────────────────────────────────────

p1, p2, p3, p4 = st.columns(4)

with p1:
    if nxt is not None:
        d_usd = nxt - cur; d_pct = d_usd / cur * 100
        sg2 = "▲" if d_usd >= 0 else "▼"; cls2 = "pos" if d_usd >= 0 else "neg"
        val = f"${nxt:.4f}"
        sub = f'<span class="{cls2}">{sg2} ${abs(d_usd):.4f} ({abs(d_pct):.2f}%)</span>'
    else:
        val = '<span style="color:#1e3a5f;font-size:.9rem">Klik Prediksi</span>'
        sub = '<span style="color:#1e3a5f">——</span>'
    st.markdown(f"""<div class="pcard">
        <div class="pcard-lbl">PREDIKSI BESOK</div>
        <div class="pcard-val" style="color:#38bdf8">{val}</div>
        <div class="pcard-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

with p2:
    if pred_high is not None:
        hp  = (pred_high - cur) / cur * 100
        val = f"${pred_high:.4f}"
        sub = f'<span class="pos">▲ +{hp:.2f}% potensi naik</span>'
    else:
        val = '<span style="color:#1e3a5f;font-size:.9rem">Klik Prediksi</span>'
        sub = '<span style="color:#1e3a5f">——</span>'
    st.markdown(f"""<div class="pcard">
        <div class="pcard-lbl">TARGET TERTINGGI</div>
        <div class="pcard-val" style="color:#34d399">{val}</div>
        <div class="pcard-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

with p3:
    if pred_low is not None:
        lp  = (pred_low - cur) / cur * 100
        val = f"${pred_low:.4f}"
        sub = f'<span class="neg">▼ {lp:.2f}% potensi turun</span>'
    else:
        val = '<span style="color:#1e3a5f;font-size:.9rem">Klik Prediksi</span>'
        sub = '<span style="color:#1e3a5f">——</span>'
    st.markdown(f"""<div class="pcard">
        <div class="pcard-lbl">TARGET TERENDAH</div>
        <div class="pcard-val" style="color:#f87171">{val}</div>
        <div class="pcard-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

with p4:
    if pred_high is not None and pred_low is not None:
        rng = pred_high - pred_low
        rp  = rng / cur * 100
        if rng == 0:
            val = f"${pred_high:.4f}"
            sub = f'<span style="color:#a78bfa">1 hari = 1 titik prediksi</span>'
        else:
            val = f"${rng:.4f}"
            sub = f'<span style="color:#a78bfa">± {rp/2:.2f}% dari skrg</span>'
    else:
        val = '<span style="color:#1e3a5f;font-size:.9rem">Klik Prediksi</span>'
        sub = '<span style="color:#1e3a5f">——</span>'
    st.markdown(f"""<div class="pcard">
        <div class="pcard-lbl">RANGE PREDIKSI</div>
        <div class="pcard-val" style="color:#a78bfa">{val}</div>
        <div class="pcard-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── LINE CHART ───────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Grafik Harga Aktual vs Prediksi</div>', unsafe_allow_html=True)
fig = line_chart(df, hist_dates, hist_pred_arr, future_dates, future_pred, close_arr)
st.plotly_chart(fig, use_container_width=True)

# ─── TABEL PREDIKSI ───────────────────────────────────────────────────────────
if future_pred is not None and len(future_pred):
    st.markdown('<div class="sec">Tabel Prediksi Harian</div>', unsafe_allow_html=True)
    rows = []
    for i, (d, p) in enumerate(zip(future_dates, future_pred)):
        d_usd = p - cur
        d_pct = d_usd / cur * 100
        rows.append({
            "Hari ke":            f"H+{i+1}",
            "Tanggal":            d.strftime("%d %b %Y"),
            "Prediksi Harga ($)": f"${p:.4f}",
            "Δ USD":              f"{'▲' if d_usd>=0 else '▼'} ${abs(d_usd):.4f}",
            "Δ (%)":              f"{'▲' if d_pct>=0 else '▼'} {abs(d_pct):.2f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
LAST UPDATE: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ·
</div>
""", unsafe_allow_html=True)