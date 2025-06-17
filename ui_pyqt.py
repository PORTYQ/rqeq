"""
PyQt6 GUI for Crypto Forecast Application
"""

import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QComboBox,
    QSpinBox,
    QCheckBox,
    QPushButton,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QProgressBar,
    QMessageBox,
    QFileDialog,
    QSplitter,
    QDoubleSpinBox,
    QRadioButton,
    QButtonGroup,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor
import pyqtgraph as pg
from pyqtgraph import PlotWidget, DateAxisItem
import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from config import config
from data import DataLoader
from features import FeatureEngineer
from model import ModelTrainer
from utils import setup_logging, calculate_metrics

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class DataLoadThread(QThread):
    """Thread for loading data without blocking UI"""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            self.progress.emit(10)
            self.status.emit("Загрузка данных криптовалюты...")

            data_loader = DataLoader()
            df = data_loader.load_data(
                source=self.params["data_source"],
                ticker=self.params["crypto"],
                history_years=self.params["history_years"],
                interval=self.params["timeframe"],
                add_macro=self.params["add_macro"],
            )

            if df is not None:
                self.progress.emit(100)
                self.status.emit("Данные загружены успешно!")
                self.finished.emit(df)
            else:
                self.error.emit("Не удалось загрузить данные")

        except Exception as e:
            self.error.emit(f"Ошибка: {str(e)}")


class ModelTrainThread(QThread):
    """Thread for training model without blocking UI"""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, data, params):
        super().__init__()
        self.data = data
        self.params = params

    def run(self):
        try:
            self.progress.emit(10)
            self.status.emit("Подготовка признаков...")

            # Feature engineering
            feature_engineer = FeatureEngineer()
            df_features = feature_engineer.prepare_features(
                self.data.copy(), scaler_type=self.params["scaler_type"]
            )

            self.progress.emit(30)
            self.status.emit("Создание обучающих окон...")

            # Create windows
            X, y, indices = feature_engineer.make_windows(
                df_features,
                feature_engineer.feature_columns,
                self.params["window_size"],
                self.params["horizon"],
                self.params["stride"],
            )

            # Scale features
            X_scaled = X.copy()
            for i in range(X.shape[0]):
                X_scaled[i] = feature_engineer.scale_features(
                    X[i], self.params["scaler_type"], fit=(i == 0)
                )

            # Split data
            train_size = int(0.8 * len(X_scaled))
            X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            self.progress.emit(50)
            self.status.emit("Обучение модели...")

            # Train model
            trainer = ModelTrainer()
            model = trainer.train_model(
                X_train,
                y_train,
                X_test,
                y_test,
                model_type=self.params["model_type"],
                use_hyperparam_search=self.params["hyper_search"],
            )

            self.progress.emit(80)
            self.status.emit("Оценка модели...")

            # Evaluate
            metrics = trainer.evaluate_model(model, X_test, y_test)
            predictions = model.predict(X_test)

            # Inverse transform
            predictions_original = feature_engineer.inverse_transform_predictions(
                predictions,
                (
                    predictions.shape[0] * predictions.shape[1],
                    len(feature_engineer.feature_columns),
                ),
            )

            results = {
                "model": model,
                "trainer": trainer,
                "feature_engineer": feature_engineer,
                "metrics": metrics,
                "predictions": predictions_original,
                "indices": indices[train_size:],
                "X_test": X_test,
                "y_test": y_test,
            }

            self.progress.emit(100)
            self.status.emit("Обучение завершено!")
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(f"Ошибка: {str(e)}")


class CandlestickWidget(FigureCanvas):
    """Widget for candlestick chart"""

    def __init__(self):
        self.fig = Figure(figsize=(10, 6))
        super().__init__(self.fig)

    def plot_candlestick(self, df):
        self.fig.clear()

        # Create subplots
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax1 = self.fig.add_subplot(gs[0])
        ax2 = self.fig.add_subplot(gs[1], sharex=ax1)

        # Candlestick data
        dates = df.index
        opens = df["Open"].values
        highs = df["High"].values
        lows = df["Low"].values
        closes = df["Close"].values

        # Colors
        up = closes >= opens
        colors = ["g" if u else "r" for u in up]

        # Plot candlesticks
        for i in range(len(dates)):
            ax1.plot([i, i], [lows[i], highs[i]], color=colors[i], linewidth=1)
            ax1.plot([i, i], [opens[i], closes[i]], color=colors[i], linewidth=3)

        ax1.set_ylabel("Цена (USD)")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-1, len(dates))

        # Volume
        if "Volume" in df.columns:
            ax2.bar(range(len(dates)), df["Volume"].values, color=colors, alpha=0.5)
            ax2.set_ylabel("Объем")
            ax2.grid(True, alpha=0.3)

        ax2.set_xlabel("Дата")
        ax2.set_xlim(-1, len(dates))

        # Format x-axis
        step = max(1, len(dates) // 10)
        ax2.set_xticks(range(0, len(dates), step))
        ax2.set_xticklabels(
            [dates[i].strftime("%Y-%m-%d") for i in range(0, len(dates), step)],
            rotation=45,
        )

        self.fig.tight_layout()
        self.draw()


class CryptoForecastApp(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.data = None
        self.results = None
        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Прогноз курса криптовалют")
        self.setGeometry(100, 100, 1400, 900)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Left panel (parameters)
        left_panel = self.create_left_panel()

        # Right panel (results)
        right_panel = self.create_right_panel()

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 1000])

        main_layout.addWidget(splitter)

        # Apply dark theme
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                background-color: #2b2b2b;
                border: 2px solid #3c3c3c;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                color: #ffffff;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
            }
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #14847f;
            }
            QPushButton:pressed {
                background-color: #0a5d61;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #3c3c3c;
                border: 1px solid #555;
                padding: 5px;
                border-radius: 3px;
            }
            QCheckBox, QRadioButton {
                spacing: 10px;
            }
            QTableWidget {
                background-color: #2b2b2b;
                gridline-color: #3c3c3c;
            }
            QProgressBar {
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0d7377;
                border-radius: 3px;
            }
        """
        )

    def create_left_panel(self):
        """Create left panel with parameters"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Data source group
        data_group = QGroupBox("Источник данных")
        data_layout = QVBoxLayout()

        self.source_group = QButtonGroup()
        self.radio_yahoo = QRadioButton("Yahoo Finance")
        self.radio_binance = QRadioButton("Binance")
        self.radio_yahoo.setChecked(True)
        self.source_group.addButton(self.radio_yahoo)
        self.source_group.addButton(self.radio_binance)

        data_layout.addWidget(self.radio_yahoo)
        data_layout.addWidget(self.radio_binance)
        data_group.setLayout(data_layout)

        # Crypto selection
        crypto_group = QGroupBox("Криптовалюта")
        crypto_layout = QVBoxLayout()

        self.crypto_combo = QComboBox()
        self.crypto_combo.addItems(config.crypto_pairs)
        crypto_layout.addWidget(self.crypto_combo)
        crypto_group.setLayout(crypto_layout)

        # Time parameters
        time_group = QGroupBox("Временные параметры")
        time_layout = QVBoxLayout()

        # Timeframe
        time_layout.addWidget(QLabel("Таймфрейм:"))
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(config.time_frames)
        time_layout.addWidget(self.timeframe_combo)

        # History
        time_layout.addWidget(QLabel("История (лет):"))
        self.history_spin = QSpinBox()
        self.history_spin.setRange(1, 5)
        self.history_spin.setValue(config.default_history_years)
        time_layout.addWidget(self.history_spin)

        time_group.setLayout(time_layout)

        # Model parameters
        model_group = QGroupBox("Параметры модели")
        model_layout = QVBoxLayout()

        # Window size
        model_layout.addWidget(QLabel("Размер окна:"))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(10, 200)
        self.window_spin.setValue(config.default_window_size)
        model_layout.addWidget(self.window_spin)

        # Horizon
        model_layout.addWidget(QLabel("Горизонт:"))
        self.horizon_spin = QSpinBox()
        self.horizon_spin.setRange(1, 30)
        self.horizon_spin.setValue(config.default_horizon)
        model_layout.addWidget(self.horizon_spin)

        # Stride
        model_layout.addWidget(QLabel("Шаг:"))
        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 10)
        self.stride_spin.setValue(config.default_stride)
        model_layout.addWidget(self.stride_spin)

        model_group.setLayout(model_layout)

        # Options
        options_group = QGroupBox("Опции")
        options_layout = QVBoxLayout()

        self.add_macro_check = QCheckBox("Добавить макроиндикаторы")
        self.add_macro_check.setChecked(True)

        self.use_volume_check = QCheckBox("Использовать реальный объем (Binance)")
        self.use_volume_check.setChecked(True)

        self.show_shap_check = QCheckBox("Показать SHAP анализ")
        self.show_shap_check.setChecked(True)

        self.hyper_search_check = QCheckBox("Поиск гиперпараметров")
        self.hyper_search_check.setChecked(True)

        options_layout.addWidget(self.add_macro_check)
        options_layout.addWidget(self.use_volume_check)
        options_layout.addWidget(self.show_shap_check)
        options_layout.addWidget(self.hyper_search_check)

        options_group.setLayout(options_layout)

        # Scaler and model type
        tech_group = QGroupBox("Технические параметры")
        tech_layout = QVBoxLayout()

        # Scaler
        tech_layout.addWidget(QLabel("Масштабирование:"))
        self.scaler_group = QButtonGroup()
        self.radio_standard = QRadioButton("Standard")
        self.radio_minmax = QRadioButton("MinMax")
        self.radio_standard.setChecked(True)
        self.scaler_group.addButton(self.radio_standard)
        self.scaler_group.addButton(self.radio_minmax)
        tech_layout.addWidget(self.radio_standard)
        tech_layout.addWidget(self.radio_minmax)

        # Model type
        tech_layout.addWidget(QLabel("Тип модели:"))
        self.model_group = QButtonGroup()
        self.radio_lstm = QRadioButton("LSTM")
        self.radio_gru = QRadioButton("GRU")
        self.radio_lstm.setChecked(True)
        self.model_group.addButton(self.radio_lstm)
        self.model_group.addButton(self.radio_gru)
        tech_layout.addWidget(self.radio_lstm)
        tech_layout.addWidget(self.radio_gru)

        tech_group.setLayout(tech_layout)

        # Buttons
        self.load_button = QPushButton("📊 Загрузить данные")
        self.load_button.clicked.connect(self.load_data)

        self.train_button = QPushButton("🤖 Обучить модель")
        self.train_button.clicked.connect(self.train_model)
        self.train_button.setEnabled(False)

        self.report_button = QPushButton("📄 Сохранить отчет")
        self.report_button.clicked.connect(self.save_report)
        self.report_button.setEnabled(False)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Готов к работе")

        # Add all to layout
        layout.addWidget(data_group)
        layout.addWidget(crypto_group)
        layout.addWidget(time_group)
        layout.addWidget(model_group)
        layout.addWidget(options_group)
        layout.addWidget(tech_group)
        layout.addWidget(self.load_button)
        layout.addWidget(self.train_button)
        layout.addWidget(self.report_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addStretch()

        return panel

    def create_right_panel(self):
        """Create right panel with results"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Tabs
        self.tabs = QTabWidget()

        # Historical data tab
        self.hist_tab = QWidget()
        hist_layout = QVBoxLayout(self.hist_tab)
        self.candlestick_widget = CandlestickWidget()
        hist_layout.addWidget(self.candlestick_widget)
        self.tabs.addTab(self.hist_tab, "📈 Исторические данные")

        # Predictions tab
        self.pred_tab = QWidget()
        pred_layout = QVBoxLayout(self.pred_tab)
        self.pred_plot = pg.PlotWidget()
        self.pred_plot.setLabel("left", "Цена (USD)")
        self.pred_plot.setLabel("bottom", "Дата")
        self.pred_plot.showGrid(x=True, y=True, alpha=0.3)
        pred_layout.addWidget(self.pred_plot)
        self.tabs.addTab(self.pred_tab, "🔮 Прогнозы")

        # Metrics tab
        self.metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(self.metrics_tab)
        self.metrics_table = QTableWidget()
        metrics_layout.addWidget(self.metrics_table)
        self.tabs.addTab(self.metrics_tab, "📊 Метрики")

        # Training history tab
        self.train_tab = QWidget()
        train_layout = QVBoxLayout(self.train_tab)
        self.train_plot = pg.PlotWidget()
        self.train_plot.setLabel("left", "Loss")
        self.train_plot.setLabel("bottom", "Эпоха")
        self.train_plot.showGrid(x=True, y=True, alpha=0.3)
        train_layout.addWidget(self.train_plot)
        self.tabs.addTab(self.train_tab, "📉 История обучения")

        # SHAP tab
        self.shap_tab = QWidget()
        shap_layout = QVBoxLayout(self.shap_tab)
        self.shap_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        shap_layout.addWidget(self.shap_canvas)
        self.tabs.addTab(self.shap_tab, "🔍 SHAP анализ")

        # Comparison tab
        self.comp_tab = QWidget()
        comp_layout = QVBoxLayout(self.comp_tab)
        self.comp_table = QTableWidget()
        comp_layout.addWidget(self.comp_table)
        self.tabs.addTab(self.comp_tab, "🔄 Сравнение моделей")

        layout.addWidget(self.tabs)

        return panel

    def get_params(self):
        """Get current parameters"""
        return {
            "data_source": (
                "Yahoo Finance" if self.radio_yahoo.isChecked() else "Binance"
            ),
            "crypto": self.crypto_combo.currentText(),
            "timeframe": self.timeframe_combo.currentText(),
            "history_years": self.history_spin.value(),
            "window_size": self.window_spin.value(),
            "horizon": self.horizon_spin.value(),
            "stride": self.stride_spin.value(),
            "add_macro": self.add_macro_check.isChecked(),
            "use_real_volume": self.use_volume_check.isChecked(),
            "show_shap": self.show_shap_check.isChecked(),
            "hyper_search": self.hyper_search_check.isChecked(),
            "scaler_type": "Standard" if self.radio_standard.isChecked() else "MinMax",
            "model_type": "LSTM" if self.radio_lstm.isChecked() else "GRU",
        }

    def load_data(self):
        """Load data in thread"""
        params = self.get_params()

        self.load_thread = DataLoadThread(params)
        self.load_thread.progress.connect(self.progress_bar.setValue)
        self.load_thread.status.connect(self.status_label.setText)
        self.load_thread.finished.connect(self.on_data_loaded)
        self.load_thread.error.connect(self.show_error)

        self.load_button.setEnabled(False)
        self.load_thread.start()

    def on_data_loaded(self, df):
        """Handle loaded data"""
        self.data = df
        self.load_button.setEnabled(True)
        self.train_button.setEnabled(True)

        # Show data info
        info = f"Загружено {len(df)} записей с {df.index[0].strftime('%Y-%m-%d')} по {df.index[-1].strftime('%Y-%m-%d')}"
        QMessageBox.information(self, "Успех", info)

        # Plot historical data
        self.candlestick_widget.plot_candlestick(df)

        # Show data statistics
        self.show_data_stats(df)

    def show_data_stats(self, df):
        """Show data statistics in metrics table"""
        stats = df.describe()

        self.metrics_table.setRowCount(len(stats.index))
        self.metrics_table.setColumnCount(len(stats.columns))
        self.metrics_table.setHorizontalHeaderLabels(stats.columns.tolist())
        self.metrics_table.setVerticalHeaderLabels(stats.index.tolist())

        for i in range(len(stats.index)):
            for j in range(len(stats.columns)):
                item = QTableWidgetItem(f"{stats.iloc[i, j]:.2f}")
                self.metrics_table.setItem(i, j, item)

    def train_model(self):
        """Train model in thread"""
        if self.data is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите данные")
            return

        params = self.get_params()

        self.train_thread = ModelTrainThread(self.data, params)
        self.train_thread.progress.connect(self.progress_bar.setValue)
        self.train_thread.status.connect(self.status_label.setText)
        self.train_thread.finished.connect(self.on_model_trained)
        self.train_thread.error.connect(self.show_error)

        self.train_button.setEnabled(False)
        self.train_thread.start()

    def on_model_trained(self, results):
        """Handle trained model"""
        self.results = results
        self.train_button.setEnabled(True)
        self.report_button.setEnabled(True)

        QMessageBox.information(self, "Успех", "Модель успешно обучена!")

        # Show metrics
        self.show_metrics(results["metrics"])

        # Plot predictions
        self.plot_predictions(results)

        # Plot training history
        if hasattr(results["trainer"], "training_history"):
            self.plot_training_history(results["trainer"].training_history)

        # SHAP analysis
        if self.show_shap_check.isChecked():
            self.show_shap_analysis(results)

        # Model comparison
        if self.add_macro_check.isChecked():
            self.compare_models(results)

    def show_metrics(self, metrics):
        """Show model metrics"""
        # Filter average metrics
        avg_metrics = {k: v for k, v in metrics.items() if "avg" in k}

        self.metrics_table.setRowCount(len(avg_metrics))
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Метрика", "Значение"])

        for i, (key, value) in enumerate(avg_metrics.items()):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(key))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))

    def plot_predictions(self, results):
        """Plot predictions"""
        self.pred_plot.clear()

        # Historical prices
        dates = self.data.index
        prices = self.data["Close"].values

        # Plot historical
        self.pred_plot.plot(range(len(dates)), prices, pen="b", name="Исторические")

        # Plot predictions
        predictions = results["predictions"]
        indices = results["indices"]

        # Show last 20 predictions
        for i in range(min(20, len(predictions))):
            idx = indices[-(i + 1)]
            if idx + results["trainer"].best_model.horizon <= len(dates):
                pred_x = list(range(idx, idx + results["trainer"].best_model.horizon))
                pred_y = predictions[-(i + 1)]
                pen = pg.mkPen(
                    color=(255, 0, 0, 100), width=2, style=Qt.PenStyle.DashLine
                )
                self.pred_plot.plot(pred_x, pred_y, pen=pen)

    def plot_training_history(self, history):
        """Plot training history"""
        self.train_plot.clear()

        if "loss" in history:
            epochs = range(len(history["loss"]))
            self.train_plot.plot(epochs, history["loss"], pen="b", name="Train Loss")

        if "val_loss" in history:
            self.train_plot.plot(epochs, history["val_loss"], pen="r", name="Val Loss")

        self.train_plot.addLegend()

    def show_shap_analysis(self, results):
        """Show SHAP analysis"""
        try:
            self.status_label.setText("Генерация SHAP анализа...")

            shap_data = results["trainer"].explain_predictions(
                results["model"],
                results["X_test"][:50],
                results["feature_engineer"].feature_columns,
                max_samples=50,
            )

            if shap_data is not None:
                # Plot SHAP summary
                ax = self.shap_canvas.figure.add_subplot(111)
                ax.clear()

                # Calculate feature importance
                feature_importance = np.abs(shap_data["shap_values"]).mean(axis=0)
                top_features_idx = np.argsort(feature_importance)[-20:]

                # Plot bar chart
                ax.barh(
                    range(len(top_features_idx)), feature_importance[top_features_idx]
                )
                ax.set_yticks(range(len(top_features_idx)))
                ax.set_yticklabels(
                    [shap_data["feature_names"][i] for i in top_features_idx]
                )
                ax.set_xlabel("SHAP важность")
                ax.set_title("Топ-20 важных признаков")

                self.shap_canvas.draw()

        except Exception as e:
            logger.error(f"SHAP analysis error: {e}")

    def compare_models(self, results):
        """Compare models with and without macro indicators"""
        try:
            comparison = results["trainer"].compare_models(
                results["X_test"],
                results["y_test"],
                results["X_test"],
                results["y_test"],
                with_macro=True,
                without_macro=True,
            )

            self.comp_table.setRowCount(len(comparison))
            self.comp_table.setColumnCount(len(comparison.columns))
            self.comp_table.setHorizontalHeaderLabels(comparison.columns.tolist())

            for i in range(len(comparison)):
                for j in range(len(comparison.columns)):
                    value = comparison.iloc[i, j]
                    if isinstance(value, (int, float)):
                        item = QTableWidgetItem(f"{value:.4f}")
                    else:
                        item = QTableWidgetItem(str(value))
                    self.comp_table.setItem(i, j, item)

        except Exception as e:
            logger.error(f"Model comparison error: {e}")

    def save_report(self):
        """Save report to file"""
        if self.results is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала обучите модель")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить отчет",
            f"crypto_forecast_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;All Files (*.*)",
        )
        if filename:
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write("=" * 80 + "\n")
                    f.write("ОТЧЕТ ПО ПРОГНОЗИРОВАНИЮ КРИПТОВАЛЮТ\n")
                    f.write("=" * 80 + "\n\n")
                    # Parameters
                    f.write("ПАРАМЕТРЫ МОДЕЛИ\n")
                    f.write("-" * 40 + "\n")
                    params = self.get_params()
                    for key, value in params.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
                    # Data info
                    f.write("ИНФОРМАЦИЯ О ДАННЫХ\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Количество записей: {len(self.data)}\n")
                    f.write(f"Период: {self.data.index[0]} - {self.data.index[-1]}\n")
                    f.write(f"Признаки: {', '.join(self.data.columns.tolist())}\n\n")
                    # Metrics
                    f.write("МЕТРИКИ МОДЕЛИ\n")
                    f.write("-" * 40 + "\n")
                    for metric, value in self.results["metrics"].items():
                        if "avg" in metric:
                            f.write(f"{metric}: {value:.4f}\n")
                    f.write("\n")
                    # Model
                    if hasattr(self.results["trainer"], "best_params"):
                        f.write("ЛУЧШИЕ ГИПЕРПАРАМЕТРЫ\n")
                        f.write("-" * 40 + "\n")
                        for key, value in self.results["trainer"].best_params.items():
                            f.write(f"{key}: {value}\n")
                QMessageBox.information(self, "Успех", f"Отчет сохранен в {filename}")
            except Exception as e:
                QMessageBox.critical(
                    self, "Ошибка", f"Ошибка сохранения отчета: {str(e)}"
                )

    def show_error(self, message):
        """Show error message"""
        QMessageBox.critical(self, "Ошибка", message)
        self.load_button.setEnabled(True)
        self.train_button.setEnabled(self.data is not None)
        self.progress_bar.setValue(0)
        self.status_label.setText("Ошибка!")


def main():
    """Main function"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = CryptoForecastApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
