# 📈 Sales Prediction Intelligence

Welcome to the **Sales Prediction Intelligence** project! This repository contains a premium, interactive web application built with [Streamlit](https://streamlit.io/) designed to help businesses optimize their advertising budgets leveraging Advanced Machine Learning.

**Test it live:** [SMARTSales AI App](https://naveen-khan-sales-prediction-app-bamzvr.streamlit.app/)

## 🚀 Overview

In a fast-paced market, allocating marketing budgets across diverse channels effectively is crucial for business growth and maximizing Return on Investment (ROI). This application bridges the gap between historical advertising expenditure (TV, Radio, Newspaper) and forecasted sales.

By utilizing **Polynomial Regression**, it allows you to simulate budget allocations and understand precisely how channel interactions act as sales drivers. 

## ✨ Key Features

- **Dynamic Machine Learning Pipeline:** Employs a robust `PolynomialFeatures` and `LinearRegression` pipeline, allowing users to capture non-linear relationships.
- **Interactive Model Tuning:** Adjust the polynomial degree (1 to 4) on the fly via the sidebar. The dashboard and models instantly react to your tuning!
- **Data-Driven Visualizations:** Features interactive Plotly visualizations demonstrating feature correlations and the true coefficient impact of variable interplay (e.g. `TV^2` vs `Radio * Newspaper`).
- **Premium User Interface:** Designed with a stunning, modern aesthetic featuring glassmorphism frosted metric cards, animated gradients, and custom sleek typography.

## 📊 Model Performance

With the default configuration (Polynomial Degree 2), the model demonstrates high predictive accuracy on the testing set:
- **R² Score:** `0.9533` (The model explains over 95% of the variance in sales)
- **Mean Squared Error (MSE):** `1.4425`

*(Note: These metrics are calculated using an 80/20 train-test split and will update dynamically within the app if you change the polynomial degree.)*

## 🛠️ Installation and Setup

To run this application locally, you will need Python installed. Follow these steps:

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd ml_projects
   ```

2. **Create a Virtual Environment (Optional but Recommended):**
   ```bash
   python -m venv .venv
   # Windows
   .venv\\Scripts\\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   Ensure you have all required packages installed by running:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```
   *This commands will automatically launch the dashboard in your default web browser.*

## 📂 Project Setup

- **`app.py`:** The main Streamlit web application.
- **`advertising.csv`:** The historical dataset containing TV, Radio, Newspaper spending, and final Sales.
- **`requirements.txt`:** All standard python package requirements to run the app.
- **`sales_Prediction based on advertisment.ipynb`:** Research notebook associated with the preliminary model explorations.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a Pull Request if you'd like to improve the UI or model implementations.

---
*Powered by Advanced Agentic Custom AI Solutions & Streamlit.*
