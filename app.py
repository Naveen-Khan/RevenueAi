ueimport streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RevenueAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0d1117;
    color: #ffffff;
}

.block-container { padding: 2rem 2.5rem !important; max-width: 1300px !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0a0f1a !important;
    border-right: 1px solid rgba(99,102,241,0.2) !important;
}

/* Brand */
.brand {
    font-family: 'Outfit', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #a78bfa, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    padding: 1rem 0 0.3rem;
}
.brand-sub {
    text-align: center;
    font-size: 0.72rem;
    color: #94a3b8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

/* KPI cards */
.kpi-row { display: flex; gap: 1rem; margin: 1.5rem 0; }
.kpi {
    flex: 1;
    background: #13192a;
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    border-top: 2px solid #6366f1;
}
.kpi-label { font-size: 0.7rem; color: #a78bfa; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; font-weight: 600; }
.kpi-val   { font-family: 'Outfit', sans-serif; font-size: 1.9rem; font-weight: 700;
             background: linear-gradient(135deg,#818cf8,#a78bfa); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.kpi-sub   { font-size: 0.75rem; color: #94a3b8; margin-top: 0.3rem; }

/* Prediction box */
.pred-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(139,92,246,0.1));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    border-top: 2px solid #818cf8;
}
.pred-tag  { font-size: 0.72rem; font-weight: 700; color: #818cf8; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 0.6rem; }
.pred-val  { font-family: 'Outfit', sans-serif; font-size: 4rem; font-weight: 800;
             background: linear-gradient(135deg,#818cf8,#a78bfa,#ec4899); -webkit-background-clip:text; -webkit-text-fill-color:transparent; line-height:1; }
.pred-unit { font-size: 0.85rem; color: #94a3b8; margin-top: 0.4rem; }

/* Insight */
.insight {
    background: rgba(99,102,241,0.07);
    border-left: 3px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: #ffffff;
    line-height: 1.6;
}

/* Section header */
.sec-head {
    font-family: 'Outfit', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 1.8rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(99,102,241,0.15);
}

/* Fit badge */
.fit-good { color: #34d399; font-weight: 700; }
.fit-warn { color: #f59e0b; font-weight: 700; }
.fit-bad  { color: #ef4444; font-weight: 700; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(15,23,42,0.6);
    padding: 5px;
    border-radius: 12px;
    border: 1px solid rgba(99,102,241,0.12);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.2rem !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color: #fff !important;
    box-shadow: 0 4px 12px rgba(99,102,241,0.3) !important;
}

/* Sidebar nav buttons */
div[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid rgba(99,102,241,0.18) !important;
    border-radius: 9px !important;
    color: #ffffff !important;
    font-weight: 500 !important;
    width: 100% !important;
    text-align: left !important;
    padding: 0.55rem 0.9rem !important;
    margin-bottom: 3px;
    transition: all 0.2s;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(99,102,241,0.1) !important;
    border-color: rgba(99,102,241,0.35) !important;
    color: #ffffff !important;
}

/* Main buttons */
div[data-testid="stMainBlockContainer"] .stButton > button {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-weight: 600 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #13192a;
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 10px;
    padding: 0.7rem 1rem !important;
}
[data-testid="stMetricLabel"] { color: #a78bfa !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-family: 'Outfit',sans-serif !important; }

hr { border-color: rgba(99,102,241,0.1) !important; margin: 1.5rem 0 !important; }
p, li { color: #ffffff !important; }
label { color: #ffffff !important; }

/* scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.35); border-radius: 99px; }
</style>

""", unsafe_allow_html=True)

# ── Chart defaults ────────────────────────────────────────────────────────────
CHART = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", size=12, color="#64748b"),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(99,102,241,0.07)",
               linecolor="rgba(99,102,241,0.15)", tickfont=dict(color="#475569")),
    yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(99,102,241,0.07)",
               linecolor="rgba(99,102,241,0.15)", tickfont=dict(color="#475569")),
)
CSCALE = [[0,"#1e1b4b"],[0.5,"#6366f1"],[1,"#f0abfc"]]

# ── Data & model ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(__file__), "advertising.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    st.error("advertising.csv not found.")
    return pd.DataFrame(columns=["TV","Radio","Newspaper","Sales"])

@st.cache_resource
def train_model(degree: int):
    df = load_data()
    if df.empty:
        return None, {}
    X, y = df[["TV","Radio","Newspaper"]], df["Sales"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=True), LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cv_r2  = cross_val_score(model, X, y, cv=5, scoring="r2").mean()
    return model, {
        "r2":     r2_score(y_test, y_pred),
        "rmse":   np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae":    mean_absolute_error(y_test, y_pred),
        "cv_r2":  cv_r2,
        "y_test": y_test.values,
        "y_pred": y_pred,
    }

def safe_predict(model, tv, radio, news):
    if model is None: return 0.0
    return float(max(0.0, model.predict([[tv, radio, news]])[0]))

def fit_badge(r2, cv_r2):
    gap = abs(r2 - cv_r2)
    if r2 >= 0.92 and gap < 0.05:  return "fit-good", "✓ Excellent fit"
    if gap > 0.10:                  return "fit-warn", "⚠ Possible overfit"
    if r2 < 0.75:                   return "fit-bad",  "⚠ Underfit"
    return "fit-good", "~ Acceptable"

# ── Session state ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "predict"

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand">🧠 RevenueAI </div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Polynomial Regression</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Navigation
    st.markdown('<p style="font-size:0.7rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;margin-bottom:6px;">Pages</p>', unsafe_allow_html=True)
    if st.button("🎯  Predict Sales",    use_container_width=True): st.session_state.page = "predict"
    if st.button("📊  Model Info",       use_container_width=True): st.session_state.page = "model"
    if st.button("📈  Data Analysis",    use_container_width=True): st.session_state.page = "analysis"

    st.markdown("---")

    # Degree
    st.markdown('<p style="font-size:0.7rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;margin-bottom:6px;">Settings</p>', unsafe_allow_html=True)
    degree = st.select_slider("Polynomial Degree", [1, 2, 3, 4], value=2,
                              help="Degree 2 gives the best balance for this dataset")

    model, metrics = train_model(degree)

    st.markdown("---")

    if metrics:
        st.markdown('<p style="font-size:0.7rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;margin-bottom:6px;">Model Stats</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("R²",   f"{metrics['r2']:.4f}",  delta=f"CV {metrics['cv_r2']:.3f}")
        c2.metric("RMSE", f"{metrics['rmse']:.2f}", delta=f"MAE {metrics['mae']:.2f}")
        cls, lbl = fit_badge(metrics["r2"], metrics["cv_r2"])
        st.markdown(f'<p style="text-align:center;margin-top:0.4rem;font-size:0.82rem;" class="{cls}">{lbl}</p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# PAGE 1 — PREDICT SALES
# ══════════════════════════════════════════════════════
if st.session_state.page == "predict":

    st.markdown('<h1 style="font-family:Outfit;font-size:2rem;font-weight:800;background:linear-gradient(135deg,#818cf8,#a78bfa,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.2rem;">🎯 Predict Sales</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#475569;margin-bottom:1.5rem;">Adjust your advertising budget and get an instant sales prediction.</p>', unsafe_allow_html=True)

    with st.expander("👋 New here? Read the Quick Start Guide", expanded=True):
        st.markdown("""
        ### Welcome to RevenueAI!
        This dashboard forecasts product sales volume based on advertising spends across three marketing channels: **TV**, **Radio**, and **Newspaper**.
        
        #### 🚀 How to Use:
        1. **Adjust Budgets:** Move the sliders below to allocate budgets (in thousands of dollars, e.g., $150k) for each channel.
        2. **Analyze Predictions:** The machine learning model instantly calculates the **Predicted Sales** (in thousands of units) and shows the budget share.
        3. **View Historical Context:** Scroll down to see how your budget scenario relates to the historical dataset on the scatter plots.
        4. **Tune Model Complexity:** Use the **Polynomial Degree** slider in the sidebar to change the mathematical complexity of the model (Degree 2 provides the best fit).
        
        #### 📂 Navigation Guide:
        - **🎯 Predict Sales (Active):** Live forecast and budget simulator.
        - **📊 Model Info:** Check prediction accuracy (R² Score, RMSE) and validation diagnostics.
        - **📈 Data Analysis:** View feature correlation heatmaps and browse the raw historical dataset.
        """)


    # Sliders
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        tv    = st.slider("📺 TV Budget ($k)",        0.0, 350.0, 150.0, 1.0)
    with c2:
        radio = st.slider("🎙️ Radio Budget ($k)",     0.0,  60.0,  30.0, 0.5)
    with c3:
        news  = st.slider("📰 Newspaper Budget ($k)", 0.0, 120.0,  40.0, 0.5)

    total      = tv + radio + news
    prediction = safe_predict(model, tv, radio, news)
    roi        = (prediction / total * 100) if total > 0 else 0

    st.markdown("---")

    col_pred, col_chart = st.columns([1, 1.1], gap="large")

    with col_pred:
        st.markdown(f"""
        <div class="pred-box">
            <div class="pred-tag">🎯 Predicted Sales</div>
            <div class="pred-val">{prediction:,.2f}</div>
            <div class="pred-unit">units (thousands)</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("📺 TV",      f"${tv:.0f}k")
        m2.metric("🎙️ Radio",  f"${radio:.0f}k")
        m3.metric("📰 News",   f"${news:.0f}k")

        st.markdown(f"""
        <div class="insight">
            💵 <b>Total Spend:</b> ${total:.1f}k &nbsp;·&nbsp;
            📈 <b>ROI:</b> {roi:.1f}%<br>
            Every $1k spent → <b>{prediction/total:.3f}k</b> sales units
        </div>
        """ if total > 0 else '<div class="insight">Set a budget to see ROI.</div>', unsafe_allow_html=True)

    with col_chart:
        if total > 0:
            pie_df = pd.DataFrame({
                "Channel": ["📺 TV", "🎙️ Radio", "📰 Newspaper"],
                "Budget":  [tv, radio, news],
            })
            fig = px.pie(pie_df, names="Channel", values="Budget",
                         color="Channel",
                         color_discrete_map={"📺 TV":"#6366f1","🎙️ Radio":"#a78bfa","📰 Newspaper":"#ec4899"},
                         hole=0.55, title="Budget Breakdown")
            fig.update_layout(**CHART, height=300,
                              legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"))
            fig.update_traces(textinfo="percent+label", textfont_size=11,
                              marker=dict(line=dict(color="#0d1117", width=2)))
            st.plotly_chart(fig, use_container_width=True, key="pred_pie")

    # Scatter on historical data
    st.markdown('<div class="sec-head">📍 Your Prediction on Historical Data</div>', unsafe_allow_html=True)

    ch_map  = {"📺 TV": ("TV", tv), "🎙️ Radio": ("Radio", radio), "📰 Newspaper": ("Newspaper", news)}
    ch_sel  = st.selectbox("Select channel", list(ch_map.keys()), key="pred_ch")
    ch_col, ch_val = ch_map[ch_sel]

    if not df.empty:
        fig_sc = px.scatter(df, x=ch_col, y="Sales", color="Sales",
                            color_continuous_scale=CSCALE, size="Sales", size_max=14,
                            opacity=0.7, title=f"Sales vs {ch_col} Spend",
                            labels={ch_col: f"{ch_col} ($k)", "Sales": "Sales (k)"},
                            hover_data={"TV":True,"Radio":True,"Newspaper":True})
        fig_sc.add_trace(go.Scatter(
            x=[ch_val], y=[prediction], mode="markers",
            marker=dict(symbol="star", size=26, color="#ec4899",
                        line=dict(width=2, color="white")),
            name="Your Prediction",
            hovertemplate=f"Spend: ${ch_val:.1f}k<br>Sales: {prediction:.2f}k<extra></extra>",
        ))
        fig_sc.update_layout(**CHART, height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_sc, use_container_width=True, key="pred_sc")


# ══════════════════════════════════════════════════════
# PAGE 2 — MODEL INFO
# ══════════════════════════════════════════════════════
elif st.session_state.page == "model":

    st.markdown('<h1 style="font-family:Outfit;font-size:2rem;font-weight:800;background:linear-gradient(135deg,#818cf8,#a78bfa,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.2rem;">📊 Model Information</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#475569;margin-bottom:1.5rem;">Performance metrics and architecture details for the active Polynomial Regression model.</p>', unsafe_allow_html=True)

    if not metrics:
        st.error("Model not loaded.")
        st.stop()

    cls, lbl = fit_badge(metrics["r2"], metrics["cv_r2"])

    # KPI row
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi">
            <div class="kpi-label">R² Score</div>
            <div class="kpi-val">{metrics['r2']:.4f}</div>
            <div class="kpi-sub">Test set</div>
        </div>
        <div class="kpi">
            <div class="kpi-label">CV R² (5-Fold)</div>
            <div class="kpi-val">{metrics['cv_r2']:.4f}</div>
            <div class="kpi-sub">Cross-validated</div>
        </div>
        <div class="kpi">
            <div class="kpi-label">RMSE</div>
            <div class="kpi-val">{metrics['rmse']:.3f}</div>
            <div class="kpi-sub">Root mean sq. error</div>
        </div>
        <div class="kpi">
            <div class="kpi-label">MAE</div>
            <div class="kpi-val">{metrics['mae']:.3f}</div>
            <div class="kpi-sub">Mean abs. error</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight">
        🏗️ <b>Model:</b> Polynomial Regression &nbsp;·&nbsp;
        <b>Degree:</b> {degree} &nbsp;·&nbsp;
        <b>Train/Test:</b> 80/20 &nbsp;·&nbsp;
        <b>CV Folds:</b> 5 &nbsp;·&nbsp;
        <b>Fit:</b> <span class="{cls}">{lbl}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Actual vs Predicted
    st.markdown('<div class="sec-head">📈 Actual vs Predicted</div>', unsafe_allow_html=True)
    lo = min(metrics["y_test"].min(), metrics["y_pred"].min()) - 0.5
    hi = max(metrics["y_test"].max(), metrics["y_pred"].max()) + 0.5

    fig_ap = go.Figure()
    fig_ap.add_trace(go.Scatter(
        x=metrics["y_test"], y=metrics["y_pred"], mode="markers",
        marker=dict(color=metrics["y_test"], colorscale=CSCALE, size=9,
                    opacity=0.8, showscale=True,
                    colorbar=dict(title="Actual", tickfont=dict(color="#475569")),
                    line=dict(width=0.5, color="#0d1117")),
        hovertemplate="Actual: %{x:.2f}k<br>Predicted: %{y:.2f}k<extra></extra>",
        name="Predictions",
    ))
    fig_ap.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="#ec4899", dash="dash", width=2),
        name="Perfect Fit", hoverinfo="skip",
    ))
    fig_ap.update_layout(**CHART, height=400,
                         title=f"Actual vs Predicted — R² = {metrics['r2']:.4f}",
                         xaxis_title="Actual Sales (k)", yaxis_title="Predicted Sales (k)")
    st.plotly_chart(fig_ap, use_container_width=True, key="model_ap")

    # Residuals
    st.markdown('<div class="sec-head">⚙️ Residual Analysis</div>', unsafe_allow_html=True)
    residuals = metrics["y_test"] - metrics["y_pred"]

    fig_res = make_subplots(rows=1, cols=2,
                            subplot_titles=["Residuals vs Predicted", "Residual Distribution"])
    fig_res.add_trace(go.Scatter(
        x=metrics["y_pred"], y=residuals, mode="markers",
        marker=dict(color="#a78bfa", size=8, opacity=0.75,
                    line=dict(width=0.5, color="#0d1117")),
        hovertemplate="Predicted: %{x:.2f}k<br>Error: %{y:.2f}k<extra></extra>",
        name="Residuals",
    ), row=1, col=1)
    fig_res.add_hline(y=0, line_dash="dash", line_color="#ec4899", line_width=2, row=1, col=1)
    fig_res.add_trace(go.Histogram(
        x=residuals, nbinsx=18, marker_color="#6366f1",
        opacity=0.8, name="Frequency",
    ), row=1, col=2)
    fig_res.update_layout(**CHART, height=380, showlegend=False)
    fig_res.update_xaxes(showgrid=True, gridcolor="rgba(99,102,241,0.07)")
    fig_res.update_yaxes(showgrid=True, gridcolor="rgba(99,102,241,0.07)")
    st.plotly_chart(fig_res, use_container_width=True, key="model_res")


# ══════════════════════════════════════════════════════
# PAGE 3 — DATA ANALYSIS
# ══════════════════════════════════════════════════════
elif st.session_state.page == "analysis":

    st.markdown('<h1 style="font-family:Outfit;font-size:2rem;font-weight:800;background:linear-gradient(135deg,#818cf8,#a78bfa,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.2rem;">📈 Data Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#475569;margin-bottom:1.5rem;">Explore the advertising dataset and understand channel relationships.</p>', unsafe_allow_html=True)

    if df.empty:
        st.error("No data loaded.")
        st.stop()

    # Dataset stats
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi">
            <div class="kpi-label">Rows</div>
            <div class="kpi-val">{len(df)}</div>
            <div class="kpi-sub">observations</div>
        </div>
        <div class="kpi">
            <div class="kpi-label">Avg Sales</div>
            <div class="kpi-val">{df['Sales'].mean():.1f}k</div>
            <div class="kpi-sub">historical mean</div>
        </div>
        <div class="kpi">
            <div class="kpi-label">Max Sales</div>
            <div class="kpi-val">{df['Sales'].max():.1f}k</div>
            <div class="kpi-sub">best record</div>
        </div>
        <div class="kpi">
            <div class="kpi-label">Min Sales</div>
            <div class="kpi-val">{df['Sales'].min():.1f}k</div>
            <div class="kpi-sub">lowest record</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📍 Scatter Plots", "🔗 Correlation", "📄 Raw Data"])

    with tab1:
        ch = st.selectbox("Select channel", ["TV","Radio","Newspaper"], key="an_ch")
        fig_sc = px.scatter(df, x=ch, y="Sales", color="Sales",
                            color_continuous_scale=CSCALE, size="Sales", size_max=14,
                            opacity=0.75, title=f"Sales vs {ch} Advertising Spend",
                            labels={ch: f"{ch} Budget ($k)", "Sales": "Sales (k)"},
                            hover_data={"TV":True,"Radio":True,"Newspaper":True})
        fig_sc.update_layout(**CHART, height=430, coloraxis_showscale=True,
                             coloraxis_colorbar=dict(title="Sales", tickfont=dict(color="#475569")))
        st.plotly_chart(fig_sc, use_container_width=True, key="an_sc")

    with tab2:
        corr = df.corr(numeric_only=True)
        fig_hm = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu",
                           title="Pearson Correlation — TV, Radio, Newspaper → Sales",
                           zmin=-1, zmax=1, aspect="auto")
        fig_hm.update_layout(**CHART, height=380)
        fig_hm.update_traces(textfont_size=14)
        st.plotly_chart(fig_hm, use_container_width=True, key="an_hm")

    with tab3:
        st.dataframe(df, use_container_width=True, height=380, key="an_raw")
        st.caption(f"📦 {len(df)} rows · 4 columns · advertising.csv")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:1rem;color:#1e293b;font-size:0.8rem;">
    <span style="font-family:Outfit;font-weight:700;background:linear-gradient(135deg,#818cf8,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        🧠 SMARTSales AI
    </span>
    &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; scikit-learn &nbsp;·&nbsp; Plotly &nbsp;·&nbsp; Polynomial Regression
</div>
""", unsafe_allow_html=True)
