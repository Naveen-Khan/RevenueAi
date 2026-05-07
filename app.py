import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Sales Dynamics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    /* Main App Customizations */
    .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Subtle enhancement to the sidebar */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.1);
        background: linear-gradient(180deg, rgba(20, 20, 20, 0.02), rgba(100, 100, 255, 0.05));
    }

    /* Glassmorphism Metric Card with Shimmer */
    .metric-card {
        background: rgba(128, 128, 128, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.1);
        border-top: 4px solid transparent;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        margin-bottom: 20px;
        color: var(--text-color);
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 50%, #00C9FF 100%);
        background-size: 200% 100%;
        animation: gradient-shift 3s ease infinite;
    }
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(128, 128, 128, 0.25);
    }

    @keyframes gradient-shift {
        0% { background-position: 100% 50%; }
        100% { background-position: -100% 50%; }
    }

    @keyframes shine {
        0% { background-position: 200% center; }
        100% { background-position: -200% center; }
    }

    .metric-value {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #6366f1 20%, #a855f7 40%, #a855f7 60%, #6366f1 80%);
        background-size: 200% auto;
        color: transparent;
        -webkit-background-clip: text;
        background-clip: text;
        animation: shine 5s linear infinite;
        line-height: 1.1;
        margin: 10px 0;
    }

    .metric-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-color);
        opacity: 0.6;
        text-transform: uppercase;
        letter-spacing: 0.15em;
    }

    .main-header {
        text-align: center;
        padding: 30px 10px;
        margin-bottom: 40px;
        background: radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.1) 0%, transparent 70%);
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Data and Model Loading ---
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), 'advertising.csv')
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.error(f"Dataset not found at {file_path}.")
        return pd.DataFrame(columns=['TV', 'Radio', 'Newspaper', 'Sales'])

@st.cache_resource
def train_model(df, degree):
    if df.empty:
        return None, 0, 0, []

    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    
    # Simple split to just get some test metrics
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    return model, r2, mse, X.columns

df = load_data()


# --- Layout: Main vs Sidebar ---
st.markdown("<div class='main-header'><h1 style='font-size: 3rem; font-weight: 800; margin-bottom: 0px;'>✨ Sales Prediction Intelligence</h1><p style='opacity: 0.7; font-size: 1.2rem; font-weight: 300; margin-top: 10px;'>Optimize your advertising budget allocations leveraging Advanced Machine Learning.</p></div>", unsafe_allow_html=True)

with st.expander("💡 Why is this Project Important?", expanded=False):
    st.markdown("""
    **Transforming Ad Spend into Predictable Revenue**
    
    In a competitive market, allocating marketing budgets effectively is crucial for business growth. This predictive application bridges the gap between historical advertising expenditure and forecasted sales. 
    
    By utilizing Advanced Machine Learning, this intelligence tool empowers you to:
    - **Optimize Return on Investment (ROI):** Understand precisely which advertising channels (TV, Radio, Newspaper) generate the most sales.
    - **Make Data-Driven Decisions:** Eliminate guesswork. Simulate various budget combinations to project revenue before committing to a financial strategy.
    - **Maximize Resource Efficiency:** Stop wasting capital on low-performing channels and redirect your funds where they have the strongest impact.
    """)

# Sidebar Inputs
st.sidebar.markdown("## 🧠 Model Tuning")
poly_degree = st.sidebar.slider("Polynomial Degree", min_value=1, max_value=4, value=2, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("## ⚙️ Budget Allocation")
st.sidebar.markdown("Adjust the slider values to simulate how different advertising budgets impact your total predicted sales.")

tv_budget = st.sidebar.slider("TV Advertising ($k)", min_value=0.0, max_value=350.0, value=150.0, step=1.0)
radio_budget = st.sidebar.slider("Radio Advertising ($k)", min_value=0.0, max_value=60.0, value=30.0, step=1.0)
news_budget = st.sidebar.slider("Newspaper Advertising ($k)", min_value=0.0, max_value=120.0, value=40.0, step=1.0)

# Train model dynamically based on degree
model, r2, mse, feature_cols = train_model(df, poly_degree)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Metrics")
st.sidebar.markdown(f"**R² Score:** {r2:.4f}")
st.sidebar.markdown(f"**MSE:** {mse:.4f}")

# --- Predictions ---
if model:
    prediction = model.predict([[tv_budget, radio_budget, news_budget]])[0]
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <div class="metric-label">Predicted Sales Performance</div>
            <div class="metric-value">{prediction:,.2f} k</div>
            <p style="margin-top: 15px; opacity: 0.5; font-size: 0.95rem; font-weight: 400;">Projected returns based on <b>${(tv_budget + radio_budget + news_budget):,.1f}k</b> total spend.</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.warning("Please ensure the data is loaded to make predictions.")

# --- Visualizations ---
st.markdown("### 🔍 Market Insights")

col_a, col_b = st.columns(2)

with col_a:
    if not df.empty:
        # Scatter Plot TV vs Sales
        fig = px.scatter(
            df, x='TV', y='Sales', 
            color='Sales', 
            color_continuous_scale=px.colors.sequential.Plasma,
            title="TV Budget vs. Sales Correlation",
            hover_data=['Radio', 'Newspaper']
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=50, b=20),
            font=dict(family="Inter", size=14)
        )
        # Add prediction point
        fig.add_trace(go.Scatter(
            x=[tv_budget], y=[prediction],
            mode='markers',
            marker=dict(color='red', size=15, symbol='star'),
            name='Current Simulation'
        ))
        st.plotly_chart(fig, use_container_width=True)

with col_b:
    if model:
        # Feature Effects (Coefficients for Polynomial Regression)
        importances = model.named_steps['linearregression'].coef_
        features = model.named_steps['polynomialfeatures'].get_feature_names_out(feature_cols)
        
        feature_df = pd.DataFrame({
            'Feature Interplay': features,
            'Coefficient Impact': np.abs(importances),
            'Raw Coefficient': importances
        }).sort_values(by='Coefficient Impact', ascending=True)
        
        if len(feature_df) > 15:
            feature_df = feature_df.tail(15)  # Keep top 15 interactions for visual clarity
            
        fig2 = px.bar(
            feature_df, x='Coefficient Impact', y='Feature Interplay', orientation='h',
            title=f"Feature Driver Influence (Degree {poly_degree})",
            color='Raw Coefficient',
            color_continuous_scale=px.colors.diverging.Tealrose,
            hover_data=['Raw Coefficient']
        )
         
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=50, b=20),
            font=dict(family="Outfit", size=14)
        )
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align: center; opacity: 0.6; font-size: 0.85rem;'>Powered by Advanced Agentic Custom AI Solutions & Streamlit</div>", unsafe_allow_html=True)
