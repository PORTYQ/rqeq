"""
Streamlit UI for Crypto Forecast Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import base64

from config import config
from data import DataLoader
from features import FeatureEngineer
from model import ModelTrainer
from utils import setup_logging, show_toast


# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def init_session_state():
    """Initialize session state variables"""
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False
    if "training_progress" not in st.session_state:
        st.session_state.training_progress = 0
    if "results" not in st.session_state:
        st.session_state.results = {}


def create_sidebar():
    """Create sidebar with parameters"""
    st.sidebar.header("Parameters")

    # Data source
    data_source = st.sidebar.radio(
        "Data source",
        config.data_sources,
        index=config.data_sources.index(config.default_source),
    )

    # Cryptocurrency
    crypto = st.sidebar.selectbox(
        "Cryptocurrency",
        config.crypto_pairs,
        index=config.crypto_pairs.index(config.default_crypto),
    )

    # Time frame
    timeframe = st.sidebar.selectbox(
        "Time frame",
        config.time_frames,
        index=config.time_frames.index(config.default_timeframe),
    )

    # History
    history_years = st.sidebar.number_input(
        "History (years)", min_value=1, max_value=5, value=config.default_history_years
    )

    # Model parameters
    st.sidebar.subheader("Model Parameters")

    window_size = st.sidebar.number_input(
        "Window size", min_value=10, max_value=200, value=config.default_window_size
    )

    horizon = st.sidebar.number_input(
        "Horizon", min_value=1, max_value=30, value=config.default_horizon
    )

    stride = st.sidebar.number_input(
        "Stride", min_value=1, max_value=10, value=config.default_stride
    )

    # Checkboxes
    st.sidebar.subheader("Options")

    add_macro = st.sidebar.checkbox("Add macro indicators", value=True)
    use_real_volume = st.sidebar.checkbox("Use real volume (Binance)", value=True)
    show_shap = st.sidebar.checkbox("Show SHAP analysis", value=True)
    hyper_search = st.sidebar.checkbox("Hyperparameter search", value=True)

    # Scaler and model type
    scaler_type = st.sidebar.radio(
        "Scaler",
        config.scaler_types,
        index=config.scaler_types.index(config.default_scaler),
    )

    model_type = st.sidebar.radio(
        "Model type",
        config.model_types,
        index=config.model_types.index(config.default_model),
    )

    return {
        "data_source": data_source,
        "crypto": crypto,
        "timeframe": timeframe,
        "history_years": history_years,
        "window_size": window_size,
        "horizon": horizon,
        "stride": stride,
        "add_macro": add_macro,
        "use_real_volume": use_real_volume,
        "show_shap": show_shap,
        "hyper_search": hyper_search,
        "scaler_type": scaler_type,
        "model_type": model_type,
    }


def plot_historical_data(df: pd.DataFrame, title: str):
    """Plot historical price data"""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3]
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    # Volume bars
    if "Volume" in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)

    fig.update_layout(title=title, height=600, xaxis_rangeslider_visible=False)

    return fig


def plot_predictions(
    df: pd.DataFrame, predictions: np.ndarray, indices: List[int], horizon: int
):
    """Plot predictions vs actual prices"""
    fig = go.Figure()

    # Historical prices
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Historical",
            line=dict(color="blue"),
        )
    )

    # Predictions
    for i, idx in enumerate(indices[-20:]):  # Show last 20 predictions
        if idx + horizon <= len(df):
            pred_dates = df.index[idx : idx + horizon]
            fig.add_trace(
                go.Scatter(
                    x=pred_dates,
                    y=predictions[i],
                    mode="lines+markers",
                    name=f"Prediction {i+1}",
                    line=dict(dash="dash"),
                    showlegend=i == 0,
                )
            )

    fig.update_layout(
        title="Price Predictions",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=500,
    )

    return fig


def plot_training_history(history: Dict[str, List[float]]):
    """Plot training history"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(len(history["loss"]))),
            y=history["loss"],
            mode="lines",
            name="Training Loss",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(len(history["val_loss"]))),
            y=history["val_loss"],
            mode="lines",
            name="Validation Loss",
        )
    )

    fig.update_layout(
        title="Training History", xaxis_title="Epoch", yaxis_title="Loss", height=400
    )

    return fig


def plot_shap_summary(shap_data: Dict):
    """Plot SHAP summary"""
    if shap_data is None:
        return None

    # Create summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_data["shap_values"],
        shap_data["X_sample"],
        feature_names=shap_data["feature_names"],
        show=False,
    )

    return fig


def plot_metrics_comparison(comparison_df: pd.DataFrame):
    """Plot metrics comparison"""
    metrics = ["MAE_avg", "RMSE_avg", "MAPE_avg"]

    fig = go.Figure()

    for metric in metrics:
        if metric in comparison_df.columns:
            fig.add_trace(
                go.Bar(
                    x=comparison_df["Model"],
                    y=comparison_df[metric],
                    name=metric.replace("_avg", ""),
                    text=comparison_df[metric].round(2),
                    textposition="auto",
                )
            )

    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Metric Value",
        barmode="group",
        height=400,
    )

    return fig


def generate_pdf_report(params: Dict, results: Dict):
    """Generate PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Title
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor("#1f77b4"),
        spaceAfter=30,
    )
    story.append(Paragraph("Cryptocurrency Price Forecast Report", title_style))
    story.append(Spacer(1, 20))

    # Parameters section
    story.append(Paragraph("Model Parameters", styles["Heading2"]))
    param_data = [
        ["Parameter", "Value"],
        ["Data Source", params["data_source"]],
        ["Cryptocurrency", params["crypto"]],
        ["Time Frame", params["timeframe"]],
        ["History", f"{params['history_years']} years"],
        ["Window Size", str(params["window_size"])],
        ["Horizon", str(params["horizon"])],
        ["Model Type", params["model_type"]],
        ["Scaler", params["scaler_type"]],
        ["Macro Indicators", "Yes" if params["add_macro"] else "No"],
    ]

    param_table = Table(param_data)
    param_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 14),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )

    story.append(param_table)
    story.append(Spacer(1, 20))

    # Results section
    if "metrics" in results:
        story.append(Paragraph("Model Performance", styles["Heading2"]))

        metrics_data = [["Metric", "Value"]]
        for metric, value in results["metrics"].items():
            if "avg" in metric:
                metrics_data.append([metric.replace("_avg", ""), f"{value:.4f}"])

        metrics_table = Table(metrics_data)
        metrics_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 14),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        story.append(metrics_table)
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


def main():
    """Main application function"""
    st.set_page_config(
        page_title="Crypto Price Forecast", page_icon="📈", layout="wide"
    )

    st.title("🚀 Cryptocurrency Price Forecast with Macro Indicators")
    st.markdown("### ML/FinTech Production-Level Application")

    # Initialize session state
    init_session_state()

    # Create sidebar
    params = create_sidebar()

    # Main content area
    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("🚀 Start / Retrain", type="primary", use_container_width=True):
            st.session_state.model_trained = False
            st.session_state.training_progress = 0

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Data loading section
    if st.button("📊 Load Data") or not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            progress_bar.progress(10)
            status_text.text("Loading cryptocurrency data...")

            # Initialize data loader
            data_loader = DataLoader()

            # Load data
            df = data_loader.load_data(
                source=params["data_source"],
                ticker=params["crypto"],
                history_years=params["history_years"],
                interval=params["timeframe"],
                add_macro=params["add_macro"],
            )

            if df is not None:
                st.session_state.data = df
                st.session_state.data_loaded = True
                progress_bar.progress(30)
                status_text.text("Data loaded successfully!")

                # Display data info
                st.info(
                    f"Loaded {len(df)} records from {df.index[0]} to {df.index[-1]}"
                )

                # Plot historical data
                with st.expander("📈 View Historical Data"):
                    fig = plot_historical_data(
                        df, f"{params['crypto']} Historical Data"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Show data sample
                with st.expander("📋 View Data Sample"):
                    st.dataframe(df.head(10))
            else:
                st.error("Failed to load data. Please check your parameters.")

    # Model training section
    if st.session_state.data_loaded and st.button("🤖 Train Model"):
        with st.spinner("Training model..."):
            progress_bar.progress(40)
            status_text.text("Preparing features...")

            # Initialize feature engineer
            feature_engineer = FeatureEngineer()

            # Prepare features
            df_features = feature_engineer.prepare_features(
                st.session_state.data.copy(), scaler_type=params["scaler_type"]
            )

            progress_bar.progress(50)
            status_text.text("Creating training windows...")

            # Create windows
            X, y, indices = feature_engineer.make_windows(
                df_features,
                feature_engineer.feature_columns,
                params["window_size"],
                params["horizon"],
                params["stride"],
            )

            # Scale features
            X_scaled = X.copy()
            for i in range(X.shape[0]):
                X_scaled[i] = feature_engineer.scale_features(
                    X[i], params["scaler_type"], fit=(i == 0)
                )

            # Split data
            train_size = int(0.8 * len(X_scaled))
            X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            progress_bar.progress(60)
            status_text.text("Training model...")

            # Initialize trainer
            trainer = ModelTrainer()

            # Train model
            model = trainer.train_model(
                X_train,
                y_train,
                X_test,
                y_test,
                model_type=params["model_type"],
                use_hyperparam_search=params["hyper_search"],
            )

            progress_bar.progress(80)
            status_text.text("Evaluating model...")

            # Evaluate model
            metrics = trainer.evaluate_model(model, X_test, y_test)

            # Make predictions
            predictions = model.predict(X_test)

            # Inverse transform predictions
            predictions_original = feature_engineer.inverse_transform_predictions(
                predictions,
                (
                    predictions.shape[0] * predictions.shape[1],
                    len(feature_engineer.feature_columns),
                ),
            )

            # Store results
            st.session_state.results = {
                "model": model,
                "trainer": trainer,
                "feature_engineer": feature_engineer,
                "metrics": metrics,
                "predictions": predictions_original,
                "indices": indices[train_size:],
                "X_test": X_test,
                "y_test": y_test,
            }
            st.session_state.model_trained = True

            progress_bar.progress(100)
            status_text.text("Model training complete!")

    # Display results
    if st.session_state.model_trained:
        st.success("✅ Model trained successfully!")

        results = st.session_state.results

        # Display metrics
        st.subheader("📊 Model Performance Metrics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("MAE", f"{results['metrics']['MAE_avg']:.4f}")
        with col2:
            st.metric("RMSE", f"{results['metrics']['RMSE_avg']:.4f}")
        with col3:
            st.metric("MAPE", f"{results['metrics']['MAPE_avg']:.2f}%")

        # Plot predictions
        st.subheader("📈 Price Predictions")
        pred_fig = plot_predictions(
            st.session_state.data,
            results["predictions"],
            results["indices"],
            params["horizon"],
        )
        st.plotly_chart(pred_fig, use_container_width=True)

        # Plot training history
        if hasattr(results["trainer"], "training_history"):
            st.subheader("📉 Training History")
            history_fig = plot_training_history(results["trainer"].training_history)
            st.plotly_chart(history_fig, use_container_width=True)

        # Model comparison (with/without macro)
        if params["add_macro"]:
            st.subheader("🔄 Impact of Macro Indicators")

            with st.spinner("Comparing models..."):
                comparison_df = results["trainer"].compare_models(
                    results["X_test"],
                    results["y_test"],
                    results["X_test"],
                    results["y_test"],
                    with_macro=True,
                    without_macro=True,
                )

                st.dataframe(comparison_df)

                # Plot comparison
                comp_fig = plot_metrics_comparison(comparison_df)
                st.plotly_chart(comp_fig, use_container_width=True)

        # SHAP analysis
        if params["show_shap"]:
            st.subheader("🔍 SHAP Feature Importance Analysis")

            with st.spinner("Generating SHAP explanations..."):
                shap_data = results["trainer"].explain_predictions(
                    results["model"],
                    results["X_test"][: config.shap_max_samples],
                    results["feature_engineer"].feature_columns,
                    max_samples=config.shap_max_samples,
                )

                if shap_data is not None:
                    # Display SHAP summary plot
                    shap_fig = plot_shap_summary(shap_data)
                    if shap_fig is not None:
                        st.pyplot(shap_fig)

                    # Display top features
                    st.subheader("🎯 Top Important Features")

                    # Calculate feature importance
                    feature_importance = np.abs(shap_data["shap_values"]).mean(axis=0)
                    top_features_idx = np.argsort(feature_importance)[-10:][::-1]

                    top_features = pd.DataFrame(
                        {
                            "Feature": [
                                shap_data["feature_names"][i] for i in top_features_idx
                            ],
                            "Importance": feature_importance[top_features_idx],
                        }
                    )

                    st.dataframe(top_features)

                    # Check macro indicator contribution
                    macro_features = [
                        f
                        for f in shap_data["feature_names"]
                        if any(macro in f for macro in ["Dollar", "Gold", "VIX", "S&P"])
                    ]
                    if macro_features:
                        macro_importance = sum(
                            feature_importance[shap_data["feature_names"].index(f)]
                            for f in macro_features
                            if f in shap_data["feature_names"]
                        )
                        total_importance = sum(feature_importance)
                        macro_percentage = (macro_importance / total_importance) * 100

                        st.info(
                            f"📊 Macro indicators contribute {macro_percentage:.1f}% to model predictions"
                        )
                else:
                    st.warning("SHAP analysis could not be completed within time limit")

        # Generate PDF report
        st.subheader("📄 Generate Report")

        if st.button("📥 Download PDF Report"):
            with st.spinner("Generating PDF report..."):
                pdf_buffer = generate_pdf_report(params, results)

                # Create download button
                st.download_button(
                    label="Download Report",
                    data=pdf_buffer,
                    file_name=f"crypto_forecast_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Cryptocurrency Price Forecast App | Built with Streamlit, TensorFlow, and SHAP</p>
            <p>Senior ML/FinTech Production Application</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
