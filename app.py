import streamlit as st
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Ford Price Estimator",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ════════════════════════════════════════════════════════════════════
#  TRAINING-TIME FEATURE STATISTICS (used to standardize inputs)
#  Source: Ford UK used-car dataset (Kaggle)
# ════════════════════════════════════════════════════════════════════
FEATURE_STATS = {
    "year":       {"mean": 2016.8,  "std": 2.3},
    "mileage":    {"mean": 23116.0, "std": 19833.0},
    "tax":        {"mean": 110.0,   "std": 83.0},
    "mpg":        {"mean": 57.0,    "std": 14.0},
    "engineSize": {"mean": 1.37,    "std": 0.5},
}

def standardize(value: float, feature: str) -> float:
    """Z-score normalize a numeric value using training-set statistics."""
    mean = FEATURE_STATS[feature]["mean"]
    std  = FEATURE_STATS[feature]["std"]
    return (value - mean) / std


# ════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&family=JetBrains+Mono:wght@400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: #080808;
    color: #e0e0e0;
    font-family: 'DM Sans', sans-serif;
}

/* Strip Streamlit chrome */
.block-container          { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }
header[data-testid="stHeader"]   { display: none; }
footer                            { display: none; }
#MainMenu                         { display: none; }

/* Scrollbar */
::-webkit-scrollbar              { width: 3px; }
::-webkit-scrollbar-track        { background: #111; }
::-webkit-scrollbar-thumb        { background: #c8102e; border-radius: 2px; }

/* ── HERO ────────────────────────────────────────── */
.hero {
    position: relative;
    width: 100%;
    padding: 3rem 4rem 2.2rem;
    border-bottom: 1px solid #161616;
    overflow: hidden;
    background: #080808;
}
.hero::after {
    content: 'FORD';
    position: absolute;
    right: -0.05em;
    top: -0.25em;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 22vw;
    color: rgba(200, 16, 46, 0.035);
    line-height: 1;
    pointer-events: none;
    user-select: none;
    letter-spacing: -0.02em;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 5px;
    color: #c8102e;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(4rem, 8vw, 7.5rem);
    line-height: 0.88;
    letter-spacing: 0.015em;
    color: #efefef;
}
.hero-title em {
    font-style: normal;
    color: #c8102e;
}
.hero-desc {
    margin-top: 1rem;
    font-size: 0.82rem;
    color: #484848;
    font-weight: 300;
    letter-spacing: 0.04em;
}

/* ── FORM PANEL ──────────────────────────────────── */
.form-wrap {
    padding: 2.8rem 2rem 2.8rem 4rem;
}

/* Section header */
.sec-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 5px;
    color: #c8102e;
    text-transform: uppercase;
    border-bottom: 1px solid #161616;
    padding-bottom: 0.55rem;
    margin-bottom: 1.1rem;
    margin-top: 2.2rem;
}
.sec-label:first-of-type { margin-top: 0.2rem; }

/* ── WIDGET OVERRIDES ────────────────────────────── */

/* Labels */
div[data-testid="stSelectbox"]  label,
div[data-testid="stNumberInput"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: #505050 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* Selectbox */
div[data-testid="stSelectbox"] > div > div {
    background: #0f0f0f !important;
    border: 1px solid #1c1c1c !important;
    border-radius: 3px !important;
    color: #d8d8d8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.93rem !important;
    transition: border-color 0.15s !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #c8102e !important;
    box-shadow: 0 0 0 2px rgba(200,16,46,0.1) !important;
}

/* Number input */
div[data-testid="stNumberInput"] input {
    background: #0f0f0f !important;
    border: 1px solid #1c1c1c !important;
    border-radius: 3px !important;
    color: #d8d8d8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.95rem !important;
    transition: border-color 0.15s !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #c8102e !important;
    box-shadow: 0 0 0 2px rgba(200,16,46,0.1) !important;
    outline: none !important;
}
div[data-testid="stNumberInput"] button {
    background: #131313 !important;
    border: 1px solid #1c1c1c !important;
    color: #444 !important;
    border-radius: 3px !important;
}
div[data-testid="stNumberInput"] button:hover {
    background: #1a1a1a !important;
    color: #ccc !important;
}

/* ── PREDICT BUTTON ──────────────────────────────── */
div.stButton > button {
    width: 100% !important;
    background: #c8102e !important;
    color: #fff !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.5rem !important;
    letter-spacing: 4px !important;
    padding: 0.75rem 1rem 0.65rem !important;
    cursor: pointer !important;
    transition: background 0.15s, transform 0.15s, box-shadow 0.15s !important;
    margin-top: 0.8rem !important;
}
div.stButton > button:hover {
    background: #a00d24 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(200,16,46,0.3) !important;
}
div.stButton > button:active { transform: translateY(0) !important; }

/* ── RESULT PANEL ────────────────────────────────── */
.result-wrap {
    padding: 2.8rem 3rem 2.8rem 0.5rem;
    position: sticky;
    top: 0;
}

.panel-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 5px;
    color: #303030;
    text-transform: uppercase;
    border-bottom: 1px solid #141414;
    padding-bottom: 0.7rem;
    margin-bottom: 1.2rem;
}

/* Idle placeholder */
.idle-box {
    border: 1px dashed #1a1a1a;
    border-radius: 4px;
    padding: 3rem 1.5rem;
    text-align: center;
}
.idle-icon { font-size: 2.4rem; display: block; margin-bottom: 0.8rem; }
.idle-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: #2e2e2e;
    letter-spacing: 3px;
    line-height: 2;
    text-transform: uppercase;
}

/* Result card */
.price-card {
    border: 1px solid #181818;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 1rem;
}
.price-card-header {
    background: #c8102e;
    padding: 0.85rem 1.4rem;
}
.price-card-header-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 4px;
    color: rgba(255,255,255,0.65);
    text-transform: uppercase;
}
.price-card-body {
    background: #0c0c0c;
    padding: 1.7rem 1.4rem 1.3rem;
}
.price-amount {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4.2rem;
    line-height: 1;
    color: #f0f0f0;
    letter-spacing: 0.01em;
}
.price-currency {
    font-size: 2.1rem;
    color: #c8102e;
    vertical-align: top;
    line-height: 1.2;
}
.price-range {
    margin-top: 0.55rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #383838;
    letter-spacing: 0.05em;
}

/* Spec chips */
.chips-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 5px;
    color: #303030;
    text-transform: uppercase;
    border-bottom: 1px solid #141414;
    padding-bottom: 0.7rem;
    margin-bottom: 0.9rem;
}
.chip-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.45rem;
}
.chip {
    background: #0c0c0c;
    border: 1px solid #181818;
    border-radius: 3px;
    padding: 0.6rem 0.85rem;
}
.chip-key {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    color: #383838;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 0.22rem;
}
.chip-val {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: #aaa;
    font-weight: 500;
}

/* Columns padding fix */
[data-testid="column"]              { padding: 0 0.4rem !important; }
[data-testid="column"]:first-child  { padding-left: 0 !important; }
[data-testid="column"]:last-child   { padding-right: 0 !important; }

/* Footer */
.footer {
    border-top: 1px solid #141414;
    padding: 1rem 4rem;
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem;
    color: #232323;
    letter-spacing: 2px;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  LOAD ARTIFACTS
# ════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    cols  = joblib.load("Car_Price_Prediction_Columns.pkl")
    model = joblib.load("Car_Price_Prediction_Model.pkl")
    return cols, model

try:
    columns, model = load_artifacts()
except FileNotFoundError:
    st.error(
        "❌ **PKL files not found.**  \n"
        "Place `Car_Price_Prediction_Columns.pkl` and `Car_Price_Prediction_Model.pkl` "
        "in the **same folder** as `app.py`, then run again."
    )
    st.stop()

# Derive option lists from column names
car_models    = sorted({c.replace("model_ ", "").replace("model_", "")
                         for c in columns if c.startswith("model_")})
transmissions = ["Automatic"] + sorted(c.replace("transmission_", "")
                                        for c in columns if c.startswith("transmission_"))
fuel_types    = ["Diesel"]    + sorted(c.replace("fuelType_", "")
                                        for c in columns if c.startswith("fuelType_"))


# ════════════════════════════════════════════════════════════════════
#  HERO
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div>
    <div class="hero-eyebrow">🚘 &nbsp; ML Price Estimator &nbsp;·&nbsp; Ford UK Dataset</div>
    <div class="hero-title">CAR <em>PRICE</em><br>ESTIMATOR</div>
    <div class="hero-desc">
      Linear Regression model &nbsp;·&nbsp;
      Fill in the vehicle details and receive an instant market valuation
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  TWO-COLUMN BODY
# ════════════════════════════════════════════════════════════════════
left, right = st.columns([2.3, 1], gap="large")

# ── LEFT: FORM ───────────────────────────────────────────────────────
with left:
    st.markdown('<div class="form-wrap">', unsafe_allow_html=True)

    # 01 — Vehicle Identity
    st.markdown('<div class="sec-label">01 &nbsp;/&nbsp; Vehicle Identity</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        default = car_models.index("Fiesta") if "Fiesta" in car_models else 0
        car_model = st.selectbox("Model", options=car_models, index=default)
    with c2:
        year = st.number_input("Year", min_value=1995, max_value=2030, value=2019, step=1)

    # 02 — Powertrain
    st.markdown('<div class="sec-label">02 &nbsp;/&nbsp; Powertrain</div>', unsafe_allow_html=True)
    c3, c4, c5 = st.columns(3)
    with c3:
        transmission = st.selectbox("Transmission", options=transmissions)
    with c4:
        fuel_type = st.selectbox("Fuel Type", options=fuel_types)
    with c5:
        engine_size = st.number_input(
            "Engine Size (L)", min_value=0.5, max_value=7.0,
            value=1.0, step=0.1, format="%.1f"
        )

    # 03 — Usage & Efficiency
    st.markdown('<div class="sec-label">03 &nbsp;/&nbsp; Usage &amp; Efficiency</div>', unsafe_allow_html=True)
    c6, c7, c8 = st.columns(3)
    with c6:
        mileage = st.number_input(
            "Mileage (miles)", min_value=0, max_value=300_000, value=20_000, step=1000
        )
    with c7:
        tax = st.number_input("Road Tax (£/yr)", min_value=0, max_value=600, value=145, step=5)
    with c8:
        mpg = st.number_input("MPG", min_value=5.0, max_value=300.0, value=55.0, step=0.5, format="%.1f")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("ESTIMATE PRICE →")
    st.markdown("</div>", unsafe_allow_html=True)


# ── RIGHT: RESULT ────────────────────────────────────────────────────
with right:
    st.markdown('<div class="result-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="panel-eyebrow">Valuation Output</div>', unsafe_allow_html=True)

    # ── Run prediction ──────────────────────────────────────────────
    if predict_clicked:
        # Build feature row — numeric features are Z-score standardized
        row = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

        row["year"]       = standardize(year,        "year")
        row["mileage"]    = standardize(mileage,     "mileage")
        row["tax"]        = standardize(tax,         "tax")
        row["mpg"]        = standardize(mpg,         "mpg")
        row["engineSize"] = standardize(engine_size, "engineSize")

        # Car model one-hot  (two naming patterns in training data)
        for candidate in (f"model_ {car_model}", f"model_{car_model}"):
            if candidate in row.columns:
                row[candidate] = 1
                break

        # Transmission one-hot (Automatic is the dropped baseline)
        if transmission != "Automatic":
            col = f"transmission_{transmission}"
            if col in row.columns:
                row[col] = 1

        # Fuel type one-hot (Diesel is the dropped baseline)
        if fuel_type != "Diesel":
            col = f"fuelType_{fuel_type}"
            if col in row.columns:
                row[col] = 1

        raw   = model.predict(row)[0]
        price = max(0.0, raw)

        st.session_state["price"]  = price
        st.session_state["inputs"] = {
            "MODEL":   car_model,
            "YEAR":    str(year),
            "GEARBOX": transmission,
            "FUEL":    fuel_type,
            "ENGINE":  f"{engine_size:.1f} L",
            "MILEAGE": f"{mileage:,} mi",
            "TAX":     f"£{tax}/yr",
            "MPG":     f"{mpg:.1f}",
        }

    # ── Display ─────────────────────────────────────────────────────
    if "price" not in st.session_state:
        st.markdown("""
        <div class="idle-box">
          <span class="idle-icon">🚘</span>
          <span class="idle-text">Fill in the form<br>and click<br>Estimate Price</span>
        </div>
        """, unsafe_allow_html=True)

    else:
        price  = st.session_state["price"]
        inputs = st.session_state["inputs"]

        if price < 500:
            st.warning("⚠️ The model returned an unusually low value for this combination. "
                       "Try adjusting year, mileage, or engine size.")
        else:
            low  = price * 0.92
            high = price * 1.08

            st.markdown(f"""
            <div class="price-card">
              <div class="price-card-header">
                <div class="price-card-header-label">Estimated Market Value</div>
              </div>
              <div class="price-card-body">
                <div class="price-amount">
                  <span class="price-currency">£</span>{price:,.0f}
                </div>
                <div class="price-range">Typical range &nbsp; £{low:,.0f} – £{high:,.0f}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="chips-label">Configuration</div>', unsafe_allow_html=True)
            chips_html = '<div class="chip-grid">'
            for k, v in inputs.items():
                chips_html += f"""
                <div class="chip">
                  <div class="chip-key">{k}</div>
                  <div class="chip-val">{v}</div>
                </div>"""
            chips_html += "</div>"
            st.markdown(chips_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
  <span>Ford Price Estimator &nbsp;·&nbsp; Linear Regression</span>
  <span>For reference only — not financial advice</span>
</div>
""", unsafe_allow_html=True)