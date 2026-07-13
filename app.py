import os
import random

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


DEFAULT_DATASET_PATH = "Students_Performance_dataset.csv"
TARGET_COLUMN = "What is your current CGPA?"


st.set_page_config(
    page_title="Student CGPA RNN Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(45, 212, 191, 0.16), transparent 30%),
            linear-gradient(135deg, #f8fafc 0%, #eef2ff 45%, #f9fafb 100%);
    }
    [data-testid="stSidebar"] {
        background: #0f172a;
    }
    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }
    .hero {
        padding: 28px 30px;
        border-radius: 18px;
        background: linear-gradient(135deg, #111827, #164e63);
        color: white;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.20);
        margin-bottom: 22px;
    }
    .hero h1 {
        font-size: 2.45rem;
        margin: 0 0 8px 0;
        letter-spacing: 0;
    }
    .hero p {
        font-size: 1.02rem;
        opacity: 0.90;
        margin: 0;
        max-width: 880px;
    }
    .metric-card {
        padding: 18px 20px;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.30);
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
    }
    .metric-card .label {
        color: #475569;
        font-size: 0.82rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .metric-card .value {
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 4px;
    }
    .result-box {
        padding: 22px;
        border-radius: 16px;
        background: #ecfeff;
        border: 1px solid #67e8f9;
        color: #164e63;
    }
    div[data-testid="stButton"] > button {
        width: 100%;
        border-radius: 12px;
        border: 0;
        color: white;
        background: linear-gradient(135deg, #0891b2, #4f46e5);
        font-weight: 750;
        padding: 0.75rem 1rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(
                {
                    "": np.nan,
                    "nan": np.nan,
                    "N": "No",
                    "no": "No",
                    "YES": "Yes",
                    "yes": "Yes",
                    "Data Schince": "Data Science",
                }
            )

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        df = df.dropna(subset=[TARGET_COLUMN])
        df = df[(df[TARGET_COLUMN] >= 0) & (df[TARGET_COLUMN] <= 4)]

    return df


@st.cache_data(show_spinner=False)
def load_default_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def detect_feature_types(X: pd.DataFrame):
    numeric_cols = []
    categorical_cols = []

    for col in X.columns:
        converted = pd.to_numeric(X[col], errors="coerce")
        numeric_ratio = converted.notna().mean()
        if numeric_ratio >= 0.85:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def coerce_numeric_columns(X: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    X = X.copy()
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def build_rnn_model(input_steps: int, units: int, dropout: float, learning_rate: float) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_steps, 1)),
            tf.keras.layers.SimpleRNN(units, activation="tanh", return_sequences=False),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_model(df: pd.DataFrame, epochs: int, batch_size: int, units: int, dropout: float, lr: float):
    set_seed(42)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    numeric_cols, categorical_cols = detect_feature_types(X_train)
    X_train = coerce_numeric_columns(X_train, numeric_cols)
    X_test = coerce_numeric_columns(X_test, numeric_cols)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    X_train_rnn = X_train_processed.reshape(X_train_processed.shape[0], X_train_processed.shape[1], 1)
    X_test_rnn = X_test_processed.reshape(X_test_processed.shape[0], X_test_processed.shape[1], 1)

    model = build_rnn_model(
        input_steps=X_train_rnn.shape[1],
        units=units,
        dropout=dropout,
        learning_rate=lr,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train_rnn,
        y_train,
        validation_split=0.20,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )

    predictions = model.predict(X_test_rnn, verbose=0).ravel()
    predictions = np.clip(predictions, 0, 4)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions, squared=False),
        "R2": r2_score(y_test, predictions),
    }

    return {
        "model": model,
        "preprocessor": preprocessor,
        "features": list(X.columns),
        "numeric_cols": numeric_cols,
        "X": X,
        "y_test": y_test.reset_index(drop=True),
        "predictions": predictions,
        "history": history.history,
        "metrics": metrics,
    }


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="hero">
        <h1>Student CGPA RNN Predictor</h1>
        <p>
            Train a Simple RNN model on your student performance dataset, inspect model quality,
            and predict a student's current CGPA from academic, lifestyle, and background details.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    use_default = st.toggle("Use default dataset path", value=True)

    st.divider()
    st.header("Training")
    epochs = st.slider("Epochs", min_value=20, max_value=250, value=90, step=10)
    batch_size = st.select_slider("Batch size", options=[8, 16, 32, 64], value=16)
    units = st.slider("RNN units", min_value=16, max_value=128, value=64, step=16)
    dropout = st.slider("Dropout", min_value=0.0, max_value=0.5, value=0.15, step=0.05)
    lr = st.select_slider("Learning rate", options=[0.0005, 0.001, 0.002, 0.005], value=0.001)


try:
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
    elif use_default:
        raw_df = load_default_dataset(DEFAULT_DATASET_PATH)
    else:
        st.info("Upload your CSV or enable the default dataset path.")
        st.stop()
except FileNotFoundError:
    st.error("Default CSV path was not found. Upload the dataset from the sidebar.")
    st.stop()
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()


df = clean_dataframe(raw_df)

if TARGET_COLUMN not in df.columns:
    st.error(f"Target column not found: {TARGET_COLUMN}")
    st.stop()

if len(df) < 30:
    st.error("The dataset is too small after cleaning. Please check missing or invalid CGPA values.")
    st.stop()


top_metrics = st.columns(4)
with top_metrics[0]:
    metric_card("Rows", f"{len(df):,}")
with top_metrics[1]:
    metric_card("Columns", f"{len(df.columns):,}")
with top_metrics[2]:
    metric_card("Average CGPA", f"{df[TARGET_COLUMN].mean():.2f}")
with top_metrics[3]:
    metric_card("CGPA Range", f"{df[TARGET_COLUMN].min():.2f} - {df[TARGET_COLUMN].max():.2f}")


tab_overview, tab_train, tab_predict = st.tabs(["Overview", "Train & Evaluate", "Predict CGPA"])


with tab_overview:
    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(25), use_container_width=True, height=430)

    with right:
        st.subheader("CGPA Distribution")
        fig = px.histogram(
            df,
            x=TARGET_COLUMN,
            nbins=25,
            color_discrete_sequence=["#0891b2"],
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Current CGPA",
            yaxis_title="Students",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.55)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Column Types")
        type_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Type": [str(df[col].dtype) for col in df.columns],
                "Missing": [int(df[col].isna().sum()) for col in df.columns],
            }
        )
        st.dataframe(type_df, use_container_width=True, height=230)


with tab_train:
    train_col, chart_col = st.columns([0.78, 1.22])

    with train_col:
        st.subheader("Model Training")
        st.write(
            "The RNN receives the processed feature vector as a sequence and learns to predict current CGPA."
        )
        train_clicked = st.button("Train RNN Model")

        if train_clicked:
            with st.spinner("Training the RNN model..."):
                st.session_state.training_result = train_model(
                    df=df,
                    epochs=epochs,
                    batch_size=batch_size,
                    units=units,
                    dropout=dropout,
                    lr=lr,
                )
            st.success("Model trained successfully.")

    if "training_result" in st.session_state:
        result = st.session_state.training_result
        metrics = result["metrics"]

        m1, m2, m3 = st.columns(3)
        with m1:
            metric_card("MAE", f"{metrics['MAE']:.3f}")
        with m2:
            metric_card("RMSE", f"{metrics['RMSE']:.3f}")
        with m3:
            metric_card("R² Score", f"{metrics['R2']:.3f}")

        with chart_col:
            history_df = pd.DataFrame(
                {
                    "Epoch": np.arange(1, len(result["history"]["loss"]) + 1),
                    "Training Loss": result["history"]["loss"],
                    "Validation Loss": result["history"]["val_loss"],
                }
            )
            loss_fig = px.line(
                history_df,
                x="Epoch",
                y=["Training Loss", "Validation Loss"],
                color_discrete_sequence=["#0891b2", "#4f46e5"],
            )
            loss_fig.update_layout(
                title="Training Curve",
                margin=dict(l=10, r=10, t=45, b=10),
                yaxis_title="MSE Loss",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.55)",
            )
            st.plotly_chart(loss_fig, use_container_width=True)

        comparison_df = pd.DataFrame(
            {
                "Actual CGPA": result["y_test"],
                "Predicted CGPA": result["predictions"],
            }
        )
        scatter_fig = px.scatter(
            comparison_df,
            x="Actual CGPA",
            y="Predicted CGPA",
            trendline="ols",
            color_discrete_sequence=["#0f766e"],
        )
        scatter_fig.update_layout(
            title="Actual vs Predicted CGPA",
            margin=dict(l=10, r=10, t=45, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.55)",
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

        st.subheader("Prediction Samples")
        st.dataframe(comparison_df.head(30), use_container_width=True)
    else:
        with chart_col:
            st.info("Click Train RNN Model to see model performance charts.")


with tab_predict:
    if "training_result" not in st.session_state:
        st.info("Train the model first, then return here to make a prediction.")
    else:
        result = st.session_state.training_result
        X_reference = result["X"]

        st.subheader("Enter Student Details")
        input_values = {}
        form_columns = st.columns(3)

        for index, col in enumerate(result["features"]):
            current_column = form_columns[index % 3]
            with current_column:
                numeric_version = pd.to_numeric(X_reference[col], errors="coerce")
                if numeric_version.notna().mean() >= 0.85:
                    min_value = float(numeric_version.min())
                    max_value = float(numeric_version.max())
                    median_value = float(numeric_version.median())
                    input_values[col] = st.number_input(
                        col,
                        min_value=min_value,
                        max_value=max_value,
                        value=median_value,
                        step=1.0 if col not in ["What was your previous SGPA?"] else 0.01,
                    )
                else:
                    options = sorted([str(v) for v in X_reference[col].dropna().unique()])
                    default_index = 0 if not options else min(len(options) - 1, 0)
                    input_values[col] = st.selectbox(col, options=options, index=default_index)

        predict_clicked = st.button("Predict Current CGPA")
        if predict_clicked:
            input_df = pd.DataFrame([input_values])
            input_df = coerce_numeric_columns(input_df, result["numeric_cols"])
            processed = result["preprocessor"].transform(input_df)
            processed_rnn = processed.reshape(processed.shape[0], processed.shape[1], 1)
            prediction = float(result["model"].predict(processed_rnn, verbose=0).ravel()[0])
            prediction = float(np.clip(prediction, 0, 4))

            st.markdown(
                f"""
                <div class="result-box">
                    <h2 style="margin: 0 0 8px 0;">Predicted Current CGPA: {prediction:.2f}</h2>
                    <p style="margin: 0;">
                        This estimate is based on the trained RNN model and the details entered above.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


st.caption(
    "Tip: For best results, keep the target column as 'What is your current CGPA?' and retrain after uploading a changed dataset."
)
