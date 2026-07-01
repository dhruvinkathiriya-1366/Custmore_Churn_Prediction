import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix

MODEL_PATH = "D:/study/Project/Custmore_churn_prediction/models/model.pkl"
X_TEST_PATH = "D:/study/Project/Custmore_churn_prediction/data/processed/x_test.csv"
Y_TEST_PATH = "D:/study/Project/Custmore_churn_prediction/data/processed/y_test.csv"
POSITIVE_LABEL = 1               # value in y_test that means "customer churned" (e.g. 1 or "Yes")
# List every feature column your model expects, IN ORDER.
# Leave as None to auto-use every column in x_test.csv.
FEATURE_COLUMNS = None

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


st.markdown("""
<div class="hero">
    <h1>📊 Customer Churn Prediction Dashboard</h1>
    <p>A simple, visual look at which customers are likely to leave — and how well the model spots them.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_test_data(path):
    return pd.read_csv(path)

model = None
X_test = None
y_true_raw = None
load_error = None

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    load_error = f"Couldn't load model from '{MODEL_PATH}': {e}"

try:
    X_test = load_test_data(X_TEST_PATH)
except Exception as e:
    load_error = (load_error + " | " if load_error else "") + \
        f"Couldn't load features from '{X_TEST_PATH}': {e}"

try:
    y_df = load_test_data(Y_TEST_PATH)
    # y_test.csv may be a single unnamed column or a named column — grab the first column either way
    y_true_raw = y_df.iloc[:, 0]
except Exception as e:
    load_error = (load_error + " | " if load_error else "") + \
        f"Couldn't load labels from '{Y_TEST_PATH}': {e}"

if load_error:
    st.error(
        "⚠️ " + load_error +
        "\n\nMake sure the paths at the top of app.py match your project folder "
        "(MODEL_PATH, X_TEST_PATH, Y_TEST_PATH)."
    )
    st.stop()

# Resolve feature columns
feature_cols = FEATURE_COLUMNS if FEATURE_COLUMNS else list(X_test.columns)
missing_cols = [c for c in feature_cols if c not in X_test.columns]
if missing_cols:
    st.error(f"⚠️ These expected feature columns are missing from x_test.csv: {missing_cols}")
    st.stop()

X_test = X_test[feature_cols]

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

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

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

left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="section-title">🥧 Who Churned vs Who Stayed</div>', unsafe_allow_html=True)
    churn_counts = pd.Series(y_true).map({1: "Churned", 0: "Stayed"}).value_counts()
    fig_pie = px.pie(
        names=churn_counts.index,
        values=churn_counts.values,
        color=churn_counts.index,
        color_discrete_map={"Churned": "#800505", "Stayed": "#00680b"},
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

st.markdown("""
<div style="text-align:center; color:#8b93ac; font-size:0.85rem; margin-top:30px;">
    Built with Streamlit · Model insights made simple for everyone 💛
</div>
""", unsafe_allow_html=True)
