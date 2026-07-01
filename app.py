"""
Customer Churn Prediction Dashboard
------------------------------------
An attractive, non-technical-friendly Streamlit app that:
  1. Loads a trained churn model (joblib/pickle)
  2. Loads a labeled test dataset (with the TRUE churn outcome)
  3. Shows a pie chart of Churned vs Stayed
  4. Shows a plain-English confusion matrix
  5. Shows Accuracy / Precision / Recall / F1 as big KPI cards
  6. Lets a non-tech user upload their own CSV or fill a form to predict a single customer

HOW TO USE:
  1. Put your saved model file next to this script and name it "model.pkl"
     (joblib.dump(model, "model.pkl"))
  2. Put a labeled test CSV next to this script named "test_data.csv".
     It must contain all the feature columns your model was trained on,
     PLUS a column with the true label. Update TARGET_COLUMN below to match.
  3. Update FEATURE_COLUMNS below to match your model's expected input columns.
  4. Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# ============================================================
# CONFIG — EDIT THESE TO MATCH YOUR PROJECT
# ============================================================
MODEL_PATH = "model.pkl"
TEST_DATA_PATH = "test_data.csv"
TARGET_COLUMN = "Churn"          # column in test_data.csv with the true answer (Yes/No or 1/0)
POSITIVE_LABEL = 1               # value in TARGET_COLUMN that means "customer churned"
# List every feature column your model expects, IN ORDER, excluding the target.
# Leave as None to auto-use every column except TARGET_COLUMN.
FEATURE_COLUMNS = None

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLE — dark navy + gold theme
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #0b132b 0%, #0f1b3c 100%);
        color: #f1f1f1;
    }

    .hero {
        background: linear-gradient(120deg, #0f1b3c 0%, #1c2e5e 100%);
        border: 1px solid #d4af37;
        border-radius: 18px;
        padding: 28px 34px;
        margin-bottom: 28px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }
    .hero h1 {
        color: #d4af37;
        margin: 0;
        font-weight: 700;
        font-size: 2.1rem;
    }
    .hero p {
        color: #cfd6e6;
        margin-top: 6px;
        font-size: 1.02rem;
    }

    .kpi-card {
        background: #131f42;
        border: 1px solid rgba(212,175,55,0.35);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.3);
        height: 100%;
    }
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: #d4af37;
        margin: 4px 0 2px 0;
    }
    .kpi-label {
        font-size: 0.95rem;
        color: #cfd6e6;
        font-weight: 600;
        letter-spacing: 0.4px;
    }
    .kpi-caption {
        font-size: 0.78rem;
        color: #8b93ac;
        margin-top: 6px;
    }

    .section-title {
        color: #d4af37;
        font-weight: 700;
        font-size: 1.3rem;
        margin: 18px 0 10px 0;
        border-left: 4px solid #d4af37;
        padding-left: 10px;
    }

    .explain-box {
        background: #131f42;
        border-radius: 14px;
        padding: 16px 20px;
        border: 1px solid rgba(212,175,55,0.25);
        color: #cfd6e6;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    [data-testid="stSidebar"] {
        background: #0b132b;
        border-right: 1px solid rgba(212,175,55,0.25);
    }

    div.stButton > button {
        background: linear-gradient(120deg, #d4af37, #b8952e);
        color: #0b132b;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 10px 22px;
    }
    div.stButton > button:hover {
        background: linear-gradient(120deg, #e8c34e, #c9a53a);
        color: #0b132b;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <h1>📊 Customer Churn Prediction Dashboard</h1>
    <p>A simple, visual look at which customers are likely to leave — and how well the model spots them.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL + DATA
# ============================================================
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_test_data(path):
    return pd.read_csv(path)

model = None
test_df = None
load_error = None

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    load_error = f"Couldn't load model from '{MODEL_PATH}': {e}"

try:
    test_df = load_test_data(TEST_DATA_PATH)
except Exception as e:
    load_error = (load_error + " | " if load_error else "") + \
        f"Couldn't load test data from '{TEST_DATA_PATH}': {e}"

if load_error:
    st.error(
        "⚠️ " + load_error +
        "\n\nMake sure `model.pkl` and `test_data.csv` are in the same folder as app.py, "
        "or edit MODEL_PATH / TEST_DATA_PATH at the top of the script."
    )
    st.stop()

# Resolve feature columns
feature_cols = FEATURE_COLUMNS if FEATURE_COLUMNS else [
    c for c in test_df.columns if c != TARGET_COLUMN
]

missing_cols = [c for c in feature_cols if c not in test_df.columns]
if TARGET_COLUMN not in test_df.columns:
    st.error(f"⚠️ Target column '{TARGET_COLUMN}' not found in test_data.csv. "
              f"Available columns: {list(test_df.columns)}")
    st.stop()
if missing_cols:
    st.error(f"⚠️ These expected feature columns are missing from test_data.csv: {missing_cols}")
    st.stop()

X_test = test_df[feature_cols]
y_true_raw = test_df[TARGET_COLUMN]

# Normalize true labels to 0/1 if they're Yes/No style
if y_true_raw.dtype == object:
    y_true = y_true_raw.map(lambda v: 1 if str(v).strip().lower() in
                             ["yes", "true", "1", "churn", "churned"] else 0)
else:
    y_true = (y_true_raw == POSITIVE_LABEL).astype(int)

# Predict
try:
    y_pred = model.predict(X_test)
    y_pred = np.array([1 if p in [1, "Yes", "yes", True] else 0 for p in y_pred])
except Exception as e:
    st.error(f"⚠️ Model prediction failed: {e}")
    st.stop()

# ============================================================
# METRICS
# ============================================================
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

# ============================================================
# KPI ROW
# ============================================================
st.markdown('<div class="section-title">🔑 Model Performance at a Glance</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
kpis = [
    (k1, "Accuracy", acc, "Out of all customers, how many were classified correctly."),
    (k2, "Precision", prec, "When the model says 'will churn', how often it's right."),
    (k3, "Recall", rec, "Out of customers who actually churned, how many the model caught."),
    (k4, "F1 Score", f1, "A balance between Precision and Recall."),
]
for col, label, value, caption in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value*100:.1f}%</div>
            <div class="kpi-caption">{caption}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# PIE CHART — Churned vs Stayed (actual distribution)
# ============================================================
left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="section-title">🥧 Who Churned vs Who Stayed</div>', unsafe_allow_html=True)
    churn_counts = pd.Series(y_true).map({1: "Churned", 0: "Stayed"}).value_counts()
    fig_pie = px.pie(
        names=churn_counts.index,
        values=churn_counts.values,
        color=churn_counts.index,
        color_discrete_map={"Churned": "#e05252", "Stayed": "#d4af37"},
        hole=0.45,
    )
    fig_pie.update_traces(
        textinfo="percent+label",
        textfont_size=15,
        marker=dict(line=dict(color="#0b132b", width=2)),
    )
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f1f1f1",
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
        margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    total = len(y_true)
    churned_n = int((y_true == 1).sum())
    stayed_n = total - churned_n
    st.markdown(f"""
    <div class="explain-box">
    Out of <b>{total}</b> customers in this dataset,
    <b style="color:#e05252;">{churned_n} ({churned_n/total*100:.1f}%)</b> churned (left) and
    <b style="color:#d4af37;">{stayed_n} ({stayed_n/total*100:.1f}%)</b> stayed.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# CONFUSION MATRIX — plain English
# ============================================================
with right:
    st.markdown('<div class="section-title">🎯 Was the Model Right?</div>', unsafe_allow_html=True)

    labels_matrix = [["Correctly said: Stayed", "Wrongly said: Will Churn"],
                      ["Wrongly said: Will Stay", "Correctly said: Churned"]]
    z = [[tn, fp], [fn, tp]]

    fig_cm = go.Figure(data=go.Heatmap(
        z=z,
        x=["Actually Stayed", "Actually Churned"],
        y=["Predicted Stayed", "Predicted Churned"],
        colorscale=[[0, "#131f42"], [1, "#d4af37"]],
        text=[[f"{labels_matrix[i][j]}<br><b>{z[i][j]}</b> customers"
               for j in range(2)] for i in range(2)],
        texttemplate="%{text}",
        textfont={"size": 13, "color": "#f1f1f1"},
        showscale=False,
        xgap=4, ygap=4,
    ))
    fig_cm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f1f1f1",
        margin=dict(t=10, b=10, l=10, r=10),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown(f"""
    <div class="explain-box">
    ✅ <b>{tp}</b> churners correctly caught &nbsp;|&nbsp;
    ✅ <b>{tn}</b> loyal customers correctly identified<br>
    ⚠️ <b>{fn}</b> churners the model <i>missed</i> (this hurts Recall) &nbsp;|&nbsp;
    ⚠️ <b>{fp}</b> loyal customers wrongly flagged as risks
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# RECALL — plain-English callout
# ============================================================
st.markdown('<div class="section-title">📌 Why Recall Matters Here</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="explain-box">
<b>Recall = {rec*100:.1f}%</b><br><br>
Think of it this way: of every <b>100 customers who were actually going to churn</b>,
this model successfully flags about <b>{rec*100:.0f} of them</b> in advance — giving the business
a chance to step in with an offer or outreach before they leave.
The remaining <b>{100-rec*100:.0f}</b> slip through undetected. A higher Recall means fewer
at-risk customers are missed, which is usually the priority in churn prevention.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# SIDEBAR — single customer prediction
# ============================================================
st.sidebar.markdown("## 🔮 Predict a Single Customer")
st.sidebar.caption("Fill in details below to check if a customer is likely to churn.")

with st.sidebar.form("predict_form"):
    input_data = {}
    for col in feature_cols:
        col_dtype = X_test[col].dtype
        if col_dtype == object or X_test[col].nunique() <= 10:
            options = sorted(X_test[col].dropna().unique().tolist())
            input_data[col] = st.selectbox(col, options)
        else:
            input_data[col] = st.number_input(
                col,
                value=float(X_test[col].median()),
            )
    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([input_data])[feature_cols]
    pred = model.predict(input_df)[0]
    is_churn = pred in [1, "Yes", "yes", True]

    try:
        proba = model.predict_proba(input_df)[0][1]
    except Exception:
        proba = None

    if is_churn:
        st.sidebar.error(f"⚠️ Likely to CHURN" + (f" ({proba*100:.1f}% risk)" if proba is not None else ""))
    else:
        st.sidebar.success(f"✅ Likely to STAY" + (f" ({(1-proba)*100:.1f}% confidence)" if proba is not None else ""))

st.markdown("""
<div style="text-align:center; color:#8b93ac; font-size:0.85rem; margin-top:30px;">
    Built with Streamlit · Model insights made simple for everyone 💛
</div>
""", unsafe_allow_html=True)
