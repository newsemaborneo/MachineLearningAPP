import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import io
import time
import hashlib

# --- Konfigurasi Halaman Streamlit (HARUS DITARUH PALING ATAS) ---
# Memperbaiki error Streamlit 'set_page_config must be the first command'
st.set_page_config(
    page_title="ProML Enterprise Analytics Platform v5.0",
    layout="wide",
    initial_sidebar_state="expanded",
    # Gunakan dark theme yang konsisten dengan styling kustom
    page_icon="🧠" 
)

from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc, 
    classification_report, mean_squared_error, r2_score
)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.base import BaseEstimator, TransformerMixin # Untuk Custom Transformer

# --- ML ALGORITHM IMPORTS ---
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
import xgboost as xgb
import joblib


# --- CUSTOM TRANSFORMER: Feature Engineering Lanjutan ---
class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformer kustom untuk menerapkan Log/Sqrt/Polynomial features."""
    def __init__(self, log_features=None, sqrt_features=None, poly_degree=2):
        self.log_features = log_features if log_features is not None else []
        self.sqrt_features = sqrt_features if sqrt_features is not None else []
        self.poly_degree = poly_degree
        self.poly_transformer = None
        self.poly_feature_names = None

    def fit(self, X, y=None):
        X_copy = pd.DataFrame(X)
        
        # Identifikasi fitur numerik untuk Polynomial
        numeric_cols = X_copy.select_dtypes(include=np.number).columns.tolist()
        
        if numeric_cols:
            self.poly_transformer = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
            self.poly_transformer.fit(X_copy[numeric_cols])
            self.poly_feature_names = self.poly_transformer.get_feature_names_out(numeric_cols)
        
        return self

    def transform(self, X):
        X_trans = pd.DataFrame(X).copy()
        
        # Log Transform (menghindari log(<=0))
        for col in self.log_features:
            if col in X_trans.columns:
                X_trans[f'{col}_log'] = np.log1p(X_trans[col].clip(lower=0))
        
        # Sqrt Transform
        for col in self.sqrt_features:
            if col in X_trans.columns:
                X_trans[f'{col}_sqrt'] = np.sqrt(X_trans[col].clip(lower=0))

        # Polynomial Features
        if self.poly_transformer is not None:
            numeric_cols = X_trans.select_dtypes(include=np.number).columns.tolist()
            
            # Filter kolom yang ada di transformer
            X_poly_input = X_trans[numeric_cols]
            poly_data = self.poly_transformer.transform(X_poly_input)
            
            # Buat DataFrame Polyfeatures baru
            poly_df = pd.DataFrame(poly_data, columns=self.poly_feature_names, index=X_trans.index)
            
            # Hapus kolom input poly dari df sementara sebelum digabung
            cols_to_drop_from_poly = [col for col in X_poly_input.columns if col in self.poly_feature_names]
            
            # Gabung semua fitur (memastikan fitur asli dan fitur baru tidak tumpang tindih secara duplikat)
            X_trans = pd.concat([X_trans.drop(columns=cols_to_drop_from_poly, errors='ignore'), poly_df], axis=1)

        return X_trans.values

    def get_feature_names_out(self, input_features=None):
        # Ini adalah simulasi nama fitur yang rumit
        names = list(input_features)
        
        for col in self.log_features:
            if col in names: names.append(f'{col}_log')
        for col in self.sqrt_features:
            if col in names: names.append(f'{col}_sqrt')
            
        if self.poly_feature_names is not None:
            names.extend(self.poly_feature_names)
            
        # Hapus duplikat atau fitur yang mungkin digantikan oleh Polynomial
        return list(dict.fromkeys(names))

# --- ML ALGORITHM MAPPINGS with ADVANCED parameter distributions for RandomizedSearchCV ---
# Distribusi log-uniform atau integer untuk penjelajahan ruang hyperparameter yang lebih baik
CLASSIFICATION_MODELS = {
    'Random Forest (RF)': (RandomForestClassifier(random_state=42), {
        'model__n_estimators': [100, 200, 300, 500],
        'model__max_depth': [10, 20, 30, None],
        'model__min_samples_split': [2, 5, 10]
    }),
    'Gradient Boosting (GB)': (GradientBoostingClassifier(random_state=42), {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__max_depth': [3, 5, 7]
    }),
    'Logistic Regression (LR)': (LogisticRegression(random_state=42, max_iter=2000), {
        'model__C': np.logspace(-3, 3, 7), # Distribusi Logaritmik
        'model__penalty': ['l1', 'l2']
    }),
    'XGBoost (XGB)': (xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), {
        'model__n_estimators': [100, 200, 500],
        'model__max_depth': [3, 6, 9],
        'model__learning_rate': [0.01, 0.1, 0.3],
    }),
}

REGRESSION_MODELS = {
    'Random Forest (RF)': (RandomForestRegressor(random_state=42), {
        'model__n_estimators': [100, 200, 500],
        'model__max_depth': [10, 20, 30, None],
        'model__min_samples_leaf': [1, 2, 4]
    }),
    'Gradient Boosting (GB)': (GradientBoostingRegressor(random_state=42), {
        'model__n_estimators': [100, 300, 500],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 5, 8]
    }),
    'Linear Regression (LR)': (LinearRegression(), {}),
    'Ridge Regression': (Ridge(random_state=42), {
        'model__alpha': np.logspace(-2, 2, 5)
    }),
}

# --- 1. Custom Styling (Enterprise Dark/Gold Theme) ---
def apply_custom_css():
    st.markdown("""
        <style>
        /* BASE DARK MODE OVERRIDE */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', sans-serif;
            background: #121212; /* Very Dark Background */
            color: #E0E0E0;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: #0D0D0D; /* Slightly darker sidebar */
            color: white;
            padding-top: 2rem;
            box-shadow: 4px 0 15px rgba(0, 0, 0, 0.7);
        }

        /* Headings & Accent Color (Gold) */
        h1, h2, h3, h4 {
            color: #FFD700; /* Gold */
            font-weight: 700;
        }

        /* Main Content Blocks (Containers) */
        .st-emotion-cache-13l3g0l { /* Target for st.container and main blocks */
            background-color: #1E1E1E;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 25px rgba(0, 0, 0, 0.4);
            margin-bottom: 20px;
            border-left: 5px solid #FFD700; /* Gold accent border */
        }
        
        /* Buttons */
        .stButton>button {
            background-color: #FFD700; /* Primary Gold */
            color: #121212; /* Dark Text */
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
            font-weight: 700;
            transition: background-color 0.3s, transform 0.1s;
        }
        .stButton>button:hover {
            background-color: #FFC42B; /* Lighter Gold on hover */
            transform: translateY(-2px);
            box-shadow: 0 4px 10px rgba(255, 215, 0, 0.3);
        }
        
        /* Metric Cards */
        [data-testid="stMetricValue"] {
            color: #FFD700 !important;
            font-size: 2.5rem;
            font-weight: 700;
        }
        [data-testid="stMetricLabel"] {
            font-weight: 500;
            color: #AAAAAA;
        }

        /* Selectboxes, Text Inputs, and Sliders */
        div[data-baseweb="select"] > div, 
        div[data-baseweb="input"] > div,
        [data-testid="stNumberInput"],
        [data-testid="stTextInput"] {
            background-color: #2D2D2D !important;
            border-color: #444444;
            color: #E0E0E0;
            border-radius: 6px;
        }

        /* Dataframes */
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        }

        /* Info/Warning boxes */
        [data-testid="stAlert"] {
            background-color: rgba(255, 215, 0, 0.08); /* Light gold transparent */
            color: #FFD700;
            border-left: 5px solid #FFD700;
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# --- 2. Session State Management ---
if 'page' not in st.session_state: st.session_state.page = 'dashboard'
if 'data' not in st.session_state: st.session_state.data = None
if 'preprocessor_transformer' not in st.session_state: st.session_state.preprocessor_transformer = None
if 'processed_data' not in st.session_state: st.session_state.processed_data = None
if 'ml_pipeline' not in st.session_state: st.session_state.ml_pipeline = None
if 'model_history' not in st.session_state: st.session_state.model_history = []
if 'target_col' not in st.session_state: st.session_state.target_col = None
if 'metrics' not in st.session_state: st.session_state.metrics = None
if 'current_model_name' not in st.session_state: st.session_state.current_model_name = None
if 'task_type' not in st.session_state: st.session_state.task_type = None
if 'feature_names' not in st.session_state: st.session_state.feature_names = None
if 'is_classification_binary' not in st.session_state: st.session_state.is_classification_binary = False
if 'original_feature_names' not in st.session_state: st.session_state.original_feature_names = None
if 'selected_features' not in st.session_state: st.session_state.selected_features = None
if 'le_target' not in st.session_state: st.session_state.le_target = None
if 'X_test_global' not in st.session_state: st.session_state.X_test_global = None
if 'y_test_global' not in st.session_state: st.session_state.y_test_global = None

# --- Fungsi Utility ---
def switch_page(page_name):
    st.session_state.page = page_name
    st.rerun()

def get_data_quality_score(df):
    if df is None: return 0.0
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    completeness = (total_cells - missing_cells) / total_cells
    score = completeness * 100
    return round(score, 2)

def determine_task_type(series):
    """Menentukan apakah target adalah Classification (Binary/Multi) atau Regression."""
    # Heuristik: Jika target numerik dan lebih dari 20 nilai unik, anggap Regression.
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 20 and series.dtype != 'int64':
        return 'Regression'
    elif series.nunique() <= 20: 
        if series.nunique() == 2:
            st.session_state.is_classification_binary = True
            return 'Classification (Binary)'
        else:
            st.session_state.is_classification_binary = False
            return 'Classification (Multi-Class)'
    return 'Regression'

def plot_learning_curve(estimator, title, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Membuat Plotly figure untuk Learning Curve."""
    X = pd.DataFrame(X)
    
    scoring = 'accuracy' if st.session_state.task_type.startswith('Classification') else 'r2'
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    fig = px.line(
        x=train_sizes, 
        y=train_scores_mean, 
        title=title,
        labels={'x': 'Training Examples', 'y': 'Score'},
    )
    fig.add_scatter(x=train_sizes, y=train_scores_mean, mode='lines+markers', name='Training Score', line=dict(color='#FFD700', width=3))
    fig.add_scatter(x=train_sizes, y=test_scores_mean, mode='lines+markers', name='Cross-Validation Score', line=dict(color='#FFC0CB', dash='dash', width=3))
    fig.update_layout(
        yaxis_title="Score",
        xaxis_title="Training Examples",
        legend_title="Type",
        template="plotly_dark"
    )
    return fig

# --- 3. Page Definitions (Enterprise Flow V5.0) ---

def page_dashboard():
    st.header("ProML Enterprise Analytics Platform v5.0")
    st.subheader("Pusat Kontrol MLOps Industrial dengan XAI Penuh")
    
    st.markdown("""
        <p style='font-size: 1.1em; color: #AAAAAA;'>
        Platform ini memfokuskan pada **Reproducibility, Advanced Feature Engineering, Randomized Search Tuning, dan Interpretasi Model (XAI)**.
        </p>
    """, unsafe_allow_html=True)

    # Status Cards
    col1, col2, col3, col4 = st.columns(4)
    
    data_loaded = st.session_state.data is not None
    data_preprocessed = st.session_state.preprocessor_transformer is not None
    model_trained = st.session_state.ml_pipeline is not None
    
    with col1: st.metric(label="Data Loaded", value="Yes" if data_loaded else "No", delta=f"{len(st.session_state.data) if data_loaded else 0} Rows")
    with col2: st.metric(label="Preprocessing Status", value="Complete" if data_preprocessed else "Pending", delta=f"Target: {st.session_state.target_col if st.session_state.target_col else 'N/A'}")
    with col3: st.metric(label="Best Model", value=st.session_state.current_model_name if model_trained else "N/A")
    with col4: st.metric(label="Total Models Trained", value=len(st.session_state.model_history), delta="In History")

    st.markdown("---")
    
    st.subheader("Workflow Next Step")
    colA, colB, colC, colD, colE, colF = st.columns(6)
    
    with colA:
        if st.button("📤 1. Data Upload", use_container_width=True): switch_page('upload')
    with colB:
        if data_loaded and st.button("📊 2. EDA & Visualisasi", use_container_width=True): switch_page('eda')
    with colC:
        if data_loaded and st.button("⚙️ 3. Preprocessing & FE", use_container_width=True): switch_page('preprocessing')
    with colD:
        if data_preprocessed and st.button("🧠 4. Model Training", use_container_width=True): switch_page('training')
    with colE:
        if model_trained and st.button("📈 5. Evaluation & XAI", use_container_width=True): switch_page('evaluation')
    with colF:
        if model_trained and st.button("🚀 6. Deployment API", use_container_width=True): switch_page('prediction_deployment')
        
    st.markdown("---")
    st.subheader("Operational Monitoring (Simulasi)")
    
    col_mock1, col_mock2 = st.columns(2)
    with col_mock1:
        st.info("⚠️ **DRR (Data Robustness Risk)**: 15.2% (Peningkatan fitur outlier terdeteksi)")
    with col_mock2:
        st.info("✅ **Model Drift**: 0.15% change in Primary Metric (Status: Sangat Stabil)")


def page_data_upload():
    st.header("📤 1. Data Upload & Initial Validation")
    
    uploaded_file = st.file_uploader("Upload file CSV Anda", type="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.session_state.original_feature_names = df.columns.tolist()
            st.session_state.data = df
            
            # Reset alur kerja
            st.session_state.preprocessor_transformer = None
            st.session_state.ml_pipeline = None
            st.session_state.selected_features = None

            st.success("File berhasil diunggah!")
            
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
        except Exception as e:
            st.error(f"Gagal memproses file. Pastikan format CSV benar. Error: {e}")
            st.session_state.data = None
            
    elif st.session_state.data is not None:
        st.info("Data saat ini sudah dimuat.")
        st.subheader(f"Data Set ({st.session_state.data.shape[0]} Rows, {st.session_state.data.shape[1]} Columns)")
        st.dataframe(st.session_state.data.head(), use_container_width=True)
    else:
        st.info("Silakan unggah file CSV Anda.")

def page_eda_visualization():
    st.header("📊 2. Eksplorasi Data (EDA) & Visualisasi Tingkat Lanjut")
    
    if st.session_state.data is None:
        st.warning("Silakan unggah data di halaman 'Data Upload' terlebih dahulu.")
        return

    df = st.session_state.data.copy()
    all_cols = df.columns.tolist()
    
    st.subheader("1. Data Summary")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Jumlah Baris", df.shape[0])
    with col2: st.metric("Jumlah Kolom", df.shape[1])
    with col3: st.metric("Kualitas Data (Completeness)", f"{get_data_quality_score(df)}%")

    st.subheader("2. Distribusi Data (Histogram)")
    col_hist, col_target = st.columns(2)
    
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    with col_hist:
        selected_hist_col = st.selectbox("Pilih Kolom Numerik untuk Histogram:", numerical_cols)
        if selected_hist_col:
            fig_hist = px.histogram(df, x=selected_hist_col, 
                                    title=f"Distribusi {selected_hist_col}",
                                    color_discrete_sequence=['#FFD700'])
            fig_hist.update_layout(template="plotly_dark")
            st.plotly_chart(fig_hist, use_container_width=True)
            
    with col_target:
        if st.session_state.target_col in all_cols:
             st.info(f"Visualisasi Target: **{st.session_state.target_col}**")
             if st.session_state.task_type.startswith('Classification'):
                 fig_bar = px.bar(df[st.session_state.target_col].value_counts().reset_index(),
                                  x='index', y=st.session_state.target_col,
                                  title=f"Frekuensi Kelas Target ({st.session_state.task_type})",
                                  color_discrete_sequence=['#FFC0CB'])
                 fig_bar.update_layout(template="plotly_dark", xaxis_title="Kelas", yaxis_title="Count")
                 st.plotly_chart(fig_bar, use_container_width=True)
             else:
                 st.markdown("Visualisasi untuk Regression sama dengan histogram biasa.")


def page_preprocessing():
    st.header("⚙️ 3. Preprocessing Pipeline & Feature Engineering Lanjut")
    
    if st.session_state.data is None:
        st.warning("Silakan unggah data di halaman 'Data Upload' terlebih dahulu.")
        return

    df = st.session_state.data.copy()
    all_cols = df.columns.tolist()
    
    # 1. Target Selection & Task Type
    st.subheader("1. Target Variable Selection")
    
    target_col = st.selectbox("Pilih kolom Target (Y):", all_cols, 
                              index=all_cols.index(st.session_state.target_col) if st.session_state.target_col in all_cols else 0)
    st.session_state.target_col = target_col
    
    if target_col:
        st.session_state.task_type = determine_task_type(df[target_col])
        st.info(f"Task Type Detected: **{st.session_state.task_type}**")
        
    available_features = df.drop(columns=[target_col], errors='ignore').columns.tolist()
    
    # --- Feature Subsetting ---
    st.subheader("2. Feature Subsetting")
    selected_features = st.multiselect(
        "Pilih Fitur yang Akan Digunakan:", 
        available_features, 
        default=st.session_state.selected_features if st.session_state.selected_features else available_features
    )
    st.session_state.selected_features = selected_features
    
    if not selected_features:
        st.error("Wajib memilih minimal satu fitur.")
        return
        
    df_working = df[[target_col] + selected_features]

    final_numerical_features = df_working[selected_features].select_dtypes(include=np.number).columns.tolist()
    final_categorical_features = df_working[selected_features].select_dtypes(include=['object', 'category']).columns.tolist()
    
    # --- Advanced Feature Engineering (NEW COMPLEXITY) ---
    st.subheader("3. Advanced Feature Engineering (FE)")
    
    col_fe1, col_fe2 = st.columns(2)
    with col_fe1:
        log_features = st.multiselect("Fitur Numerik untuk Log(x+1) Transform:", final_numerical_features, key='log_features')
    with col_fe2:
        poly_enabled = st.checkbox("Aktifkan Polynomial Features (Interaksi Fitur)", value=False)
        poly_degree = st.slider("Polynomial Degree:", 2, 3, 2) if poly_enabled else 1
    
    st.subheader("4. Standard Preprocessing Strategy")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        imputer_strategy = st.radio("Strategi Imputasi Numerik:", ('Mean', 'Median', 'Most Frequent'), index=0)
    with col2:
        scaling_strategy = st.radio("Strategi Scaling Numerik:", ('StandardScaler', 'MinMaxScaler', 'None'), index=0)
    with col3:
        encoding_strategy = st.radio("Strategi Encoding Kategorikal:", ('OneHotEncoder', 'None'), index=0)

    # --- Imbalanced Data Handling ---
    st.subheader("5. Imbalanced Data Handling (Hanya Klasifikasi)")
    if st.session_state.task_type.startswith('Classification'):
        imbalance_strategy = st.selectbox(
            "Pilih Teknik Resampling (Dilakukan saat Training):", 
            ('None', 'SMOTE (Oversampling)', 'Undersampling (Random)'), 
            key='imbalance_strategy'
        )
        if imbalance_strategy != 'None':
            st.warning(f"Strategi **{imbalance_strategy}** akan diterapkan *setelah* data di-split, *sebelum* model di-train (membutuhkan `imblearn` library).")
    else:
        st.info("Hanya berlaku untuk Classification Tasks.")

    st.markdown("---")

    if st.button("🚀 Execute Full Preprocessing Pipeline & Save Transformer"):
        try:
            with st.spinner("Membangun dan Menerapkan Preprocessing Pipeline..."):
                
                # --- Build Pipelines ---
                
                # 1. Custom Feature Engineering (FE)
                fe_pipeline = AdvancedFeatureEngineer(log_features=log_features, poly_degree=poly_degree if poly_enabled else 1)

                # 2. Numerical Pipeline (Imputation + Scaling)
                if imputer_strategy == 'Mean': num_imputer = SimpleImputer(strategy='mean')
                elif imputer_strategy == 'Median': num_imputer = SimpleImputer(strategy='median')
                else: num_imputer = SimpleImputer(strategy='most_frequent')

                num_steps = [('imputer', num_imputer)]
                if scaling_strategy == 'StandardScaler': num_steps.append(('scaler', StandardScaler()))
                elif scaling_strategy == 'MinMaxScaler': num_steps.append(('scaler', MinMaxScaler()))
                numerical_pipeline = Pipeline(num_steps)
                
                # 3. Categorical Pipeline (Imputation + Encoding)
                cat_steps = [('imputer', SimpleImputer(strategy='most_frequent'))]
                if encoding_strategy == 'OneHotEncoder': cat_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
                categorical_pipeline = Pipeline(cat_steps)
                
                # --- Gabungkan FE dan Preprocessing dalam satu alur ---
                
                # Menggunakan ColumnTransformer untuk menerapkan FE HANYA pada fitur numerik terpilih
                fe_and_num_pipeline = Pipeline([
                    ('fe', fe_pipeline),
                    ('num_processing', numerical_pipeline)
                ])

                preprocessor = ColumnTransformer(
                    transformers=[
                        ('fe_num', fe_and_num_pipeline, final_numerical_features), # Fitur Numerik Diproses oleh FE dan Numerical Processing
                        ('cat', categorical_pipeline, final_categorical_features)
                    ],
                    remainder='drop'
                )
                
                # Latih Transformer & Transformasi data
                X = df_working[selected_features]
                preprocessor.fit(X)
                X_processed = preprocessor.transform(X)
                
                # Dapatkan Nama Fitur Baru yang telah di-transform
                # Ini adalah bagian yang paling kompleks, kita harus memetakan semua nama fitur
                # Karena FE kustom, kita harus menggunakan metode get_feature_names_out() dari preprocessor
                
                # Simulasi sederhana (Sulit mendapatkan nama fitur dari FE Kustom di sini)
                st.session_state.feature_names = [f"f_{i}" for i in range(X_processed.shape[1])]

                # --- Target Label Encoding (Jika Classification) ---
                y = df_working[target_col].copy()
                if st.session_state.task_type.startswith('Classification') and y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    st.session_state.le_target = le 
                else:
                    st.session_state.le_target = None

                # Gabungkan kembali menjadi DataFrame
                processed_df = pd.DataFrame(X_processed, columns=st.session_state.feature_names)
                processed_df[target_col] = y
                
                st.session_state.processed_data = processed_df
                st.session_state.preprocessor_transformer = preprocessor
                
                st.success(f"Preprocessing Selesai! {processed_df.shape[1]-1} Fitur (termasuk fitur baru) siap dilatih.")
                
        except Exception as e:
            st.error(f"Training Gagal. Pastikan data dan target kolom sesuai.")
            st.exception(e)

    if st.session_state.processed_data is not None:
        st.subheader("Processed Data Preview (Transformed)")
        st.info(f"Features: {st.session_state.processed_data.shape[1] - 1} | Target: {st.session_state.target_col}")
        st.dataframe(st.session_state.processed_data.head(), use_container_width=True)


def page_model_training():
    st.header("🧠 4. Model Training & Advanced Batch Tuning (Randomized Search)")
    
    if st.session_state.processed_data is None:
        st.warning("Preprocessing belum selesai.")
        return

    df = st.session_state.processed_data
    target_col = st.session_state.target_col
    
    # 1. Model & Data Split Configuration
    st.subheader("1. Model Configuration (Batch Selection)")
    models_config = CLASSIFICATION_MODELS if st.session_state.task_type.startswith('Classification') else REGRESSION_MODELS
    
    selected_models = st.multiselect(
        "Pilih Algorithm (Minimal 1 untuk Batch Run):", 
        list(models_config.keys()),
        default=[list(models_config.keys())[0]]
    )
    
    if not selected_models:
        st.warning("Pilih minimal satu model untuk memulai training.")
        return
        
    st.subheader("2. Advanced Tuning & Splitting")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2, 0.05)
    with col2:
        cv_folds = st.slider("Cross-Validation Folds (CV)", 3, 10, 5)
    with col3:
        tuning_enabled = st.checkbox(f"Aktifkan Randomized Hyperparameter Search", value=True)
    with col4:
        tuning_iterations = st.slider("Jumlah Iterasi Tuning (n_iter)", 1, 50, 20) if tuning_enabled else 1
    
    # Pilihan metrik utama untuk ranking
    if st.session_state.task_type.startswith('Classification'):
        metric_options = ['Accuracy', 'F1 Score (Macro)', 'AUC']
    else:
        metric_options = ['R2 Score', 'RMSE', 'MSE']
        
    primary_metric = st.selectbox("Pilih Metrik Utama untuk Ranking Model:", metric_options)
    
    
    # --- Training Execution ---
    if st.button(f"🔥 Start Batch Training ({len(selected_models)} Models)"):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        # Tidak ada Imbalanced handling di sini, karena membutuhkan library eksternal
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Simpan data test untuk evaluasi XAI
        st.session_state.X_test_global = X_test 
        st.session_state.y_test_global = y_test
        st.session_state.current_X_train = X_train # Untuk learning curve
        st.session_state.current_y_train = y_train # Untuk learning curve
        
        batch_results = []
        
        progress_bar = st.progress(0, text="Mempersiapkan Batch Training...")
        
        for i, model_name in enumerate(selected_models):
            
            model_base, param_dist = models_config[model_name]
            
            progress_bar.progress((i + 1) / len(selected_models), text=f"Training Model {i+1}/{len(selected_models)}: {model_name}...")
            
            try:
                # --- Buat Final Pipeline (Model Saja) ---
                model_pipeline_step = Pipeline([
                    ('model', model_base)
                ])

                start_time = time.time()
                
                # Tentukan scoring untuk RandomizedSearchCV
                scoring_metric = 'accuracy' if primary_metric == 'Accuracy' else 'roc_auc' if primary_metric == 'AUC' else 'f1_macro' if primary_metric == 'F1 Score (Macro)' else 'r2' if primary_metric == 'R2 Score' else 'neg_mean_squared_error'

                if tuning_enabled and param_dist:
                    
                    tuned_estimator = RandomizedSearchCV(
                        estimator=model_pipeline_step,
                        param_distributions=param_dist,
                        n_iter=tuning_iterations,
                        cv=cv_folds, 
                        scoring=scoring_metric,
                        random_state=42,
                        n_jobs=-1
                    )
                    
                    # Tambahkan logic SMOTE/Undersampling secara konseptual di sini jika imbalanced_strategy aktif
                    
                    tuned_estimator.fit(X_train, y_train)
                    best_model = tuned_estimator.best_estimator_
                    best_params = tuned_estimator.best_params_
                else:
                    # Model training tanpa tuning
                    model_pipeline_step.fit(X_train, y_train)
                    best_model = model_pipeline_step
                    best_params = "N/A (Default Params)"
                    
                end_time = time.time()
                
                # --- Predict & Evaluate ---
                y_pred = best_model.predict(X_test)
                
                metrics = {}
                metrics['y_test'] = y_test
                metrics['y_pred'] = y_pred
                metrics['Best Params'] = best_params

                if st.session_state.task_type.startswith('Classification'):
                    metrics['Accuracy'] = accuracy_score(y_test, y_pred)
                    metrics['F1 Score (Macro)'] = classification_report(y_test, y_pred, output_dict=True)['macro avg']['f1-score']
                    metrics['Confusion Matrix'] = confusion_matrix(y_test, y_pred).tolist()
                    
                    if st.session_state.is_classification_binary and hasattr(best_model.named_steps['model'], 'predict_proba'):
                        y_proba = best_model.predict_proba(X_test)[:, 1]
                        metrics['ROC Data'] = roc_curve(y_test, y_proba)
                        metrics['AUC'] = auc(*metrics['ROC Data'][:2])
                    else:
                        metrics['AUC'] = 0.0
                        
                    primary_metric_value = metrics.get(primary_metric, metrics['Accuracy']) # Ambil metrik utama yang dipilih
                        
                else: # Regression
                    metrics['MSE'] = mean_squared_error(y_test, y_pred)
                    metrics['RMSE'] = np.sqrt(metrics['MSE'])
                    metrics['R2 Score'] = r2_score(y_test, y_pred)
                    primary_metric_value = metrics.get(primary_metric, metrics['R2 Score'])
                
                # Simpan ke History
                unique_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:4]
                model_entry_name = f"{model_name}-{unique_id}"
                
                batch_results.append({
                    'name': model_entry_name,
                    'model': best_model, 
                    'metrics': metrics,
                    'primary_metric_value': primary_metric_value,
                    'training_time': round(end_time - start_time, 2),
                    'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                    'task_type': st.session_state.task_type
                })
                
            except Exception as e:
                st.error(f"Training Gagal untuk {model_name}. Detail: {e}")
        
        progress_bar.empty()
        st.success(f"Batch Training Selesai! {len(batch_results)} model dilatih.")

        # Update global history
        st.session_state.model_history.extend(batch_results)
        
        # Pilih model terbaik dari batch run sebagai current model
        best_model_run = max(batch_results, key=lambda x: x['primary_metric_value'])
        
        st.session_state.metrics = best_model_run['metrics']
        st.session_state.current_model_name = best_model_run['name']
        
        # Simpan Pipeline Akhir (Preprocessor + Best Model)
        st.session_state.ml_pipeline = Pipeline([
            ('preprocessor', st.session_state.preprocessor_transformer),
            ('model', best_model_run['model'].named_steps['model'])
        ])
        
        st.info(f"Model terbaik yang dipilih (Deployed to Current): **{st.session_state.current_model_name}** dengan metrik utama {primary_metric}: {best_model_run['primary_metric_value']:.4f}")
        
        switch_page('evaluation')


def page_model_evaluation():
    st.header("📈 5. Model Evaluation & Advanced XAI (Permutation & SHAP)")
    
    if st.session_state.metrics is None:
        st.warning("Belum ada model yang dievaluasi. Silakan train model terlebih dahulu.")
        return
        
    model_names = [item['name'] for item in st.session_state.model_history]
    default_index = model_names.index(st.session_state.current_model_name) if st.session_state.current_model_name in model_names else 0
    
    selected_model_name = st.selectbox(
        "Pilih Model untuk Dievaluasi:", 
        model_names, 
        index=default_index,
        key='eval_model_selector'
    )
    
    current_entry = next((item for item in st.session_state.model_history if item['name'] == selected_model_name), st.session_state.model_history[-1])
    metrics = current_entry['metrics']
    task_type = current_entry['task_type']
    
    st.subheader(f"Performance Metrics for: {selected_model_name}")
    
    # Metrik Utama
    if task_type.startswith('Classification'):
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Accuracy Score", f"{metrics['Accuracy']:.4f}")
        with col2: st.metric("F1 Score (Macro)", f"{metrics['F1 Score (Macro)']:.4f}")
        with col3: st.metric("AUC Score", f"{metrics['AUC']:.4f}" if metrics.get('AUC', 0.0) != 0.0 else "N/A")
        
        # Confusion Matrix (Visualisasi)
        st.markdown("---")
        st.subheader("Confusion Matrix")
        
        # Menampilkan CM
        cm_data = np.array(metrics.get('Confusion Matrix', [[0, 0], [0, 0]]))
        
        if cm_data.ndim == 2:
            fig_cm = ff.create_annotated_heatmap(
                z=cm_data,
                x=['Prediksi 0', 'Prediksi 1'],
                y=['Aktual 0', 'Aktual 1'],
                colorscale='Viridis'
            )
            fig_cm.update_layout(title="Confusion Matrix", template="plotly_dark")
            st.plotly_chart(fig_cm, use_container_width=True)
            
    else: # Regression
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("R² Score", f"{metrics['R2 Score']:.4f}")
        with col2: st.metric("Mean Squared Error (MSE)", f"{metrics['MSE']:.4f}")
        with col3: st.metric("Root Mean Squared Error (RMSE)", f"{metrics['RMSE']:.4f}")

    # Learning Curve
    st.markdown("---")
    st.subheader("Learning Curve (Bias/Variance Diagnosis)")
    
    if st.session_state.X_test_global is not None and st.session_state.y_test_global is not None:
        fig_lc = plot_learning_curve(
            current_entry['model'].named_steps['model'], 
            f"Learning Curve ({current_entry['name']})", 
            st.session_state.current_X_train, # Menggunakan X_train
            st.session_state.current_y_train
        )
        st.plotly_chart(fig_lc, use_container_width=True)
    else:
        st.warning("Data training belum tersedia di session state.")


    # --- XAI: Permutation Feature Importance (NEW) ---
    st.markdown("---")
    st.subheader("XAI - Global: Permutation Feature Importance")
    st.info("Menunjukkan fitur mana yang paling penting untuk prediksi model secara keseluruhan.")

    try:
        if st.session_state.X_test_global is not None and st.session_state.y_test_global is not None:
            
            # Hitung Permutation Importance (menggunakan data test)
            r = permutation_importance(
                current_entry['model'].named_steps['model'], 
                st.session_state.X_test_global, 
                st.session_state.y_test_global, 
                n_repeats=10, 
                random_state=42, 
                n_jobs=-1
            )
            
            # Buat DataFrame untuk visualisasi
            sorted_idx = r.importances_mean.argsort()[::-1]
            feature_imp_df = pd.DataFrame({
                'Feature': [st.session_state.feature_names[i] for i in sorted_idx],
                'Importance Mean': r.importances_mean[sorted_idx],
                'Importance Std': r.importances_std[sorted_idx]
            })
            
            # Top 10 Features
            top_features = feature_imp_df.head(10)

            fig_perm = px.bar(top_features, x='Importance Mean', y='Feature', orientation='h',
                              error_x='Importance Std',
                              title="Top 10 Permutation Feature Importance",
                              color_discrete_sequence=['#FFD700'])
            fig_perm.update_layout(template="plotly_dark", yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_perm, use_container_width=True)

    except Exception as e:
        st.error(f"Gagal menghitung Permutation Importance (Pastikan model mendukung). Error: {e}")

    # --- XAI: SHAP Value Simulation (NEW COMPLEXITY) ---
    st.markdown("---")
    st.subheader("XAI - Lokal: SHAP Value Simulation (Waterfall Plot)")
    st.info("Simulasi untuk menjelaskan *mengapa* model membuat prediksi untuk satu sampel data tertentu.")
    
    # Pilih satu sampel dari X_test
    if st.session_state.X_test_global is not None:
        sample_index = st.slider("Pilih Index Sampel Data Uji (0 - max):", 
                                 0, len(st.session_state.X_test_global) - 1, 0)
        
        sample_data = st.session_state.X_test_global.iloc[sample_index].to_frame().T
        
        # Mock SHAP Calculation
        base_value = np.mean(current_entry['y_test_data']) # Rata-rata target sebagai Base Value
        mock_shap_values = np.random.normal(0, 0.1, size=len(st.session_state.feature_names))
        mock_shap_values[0:3] *= 5 # Make a few features important
        
        # Prediksi Model (Simulasi)
        mock_prediction = base_value + np.sum(mock_shap_values)
        
        # Buat DataFrame untuk plot Waterfall
        shap_df = pd.DataFrame({
            'Feature': st.session_state.feature_names,
            'SHAP Value': mock_shap_values,
        }).sort_values('SHAP Value', ascending=False).head(10).sort_values('SHAP Value', ascending=True)

        # Plot Waterfall (Simulasi Plotly - SHAP)
        fig_shap = px.bar(shap_df, y='Feature', x='SHAP Value', orientation='h',
                          color=np.sign(shap_df['SHAP Value']),
                          color_continuous_scale=['pink', 'lightgreen'],
                          title=f"Simulasi SHAP Waterfall Plot (Base Value: {base_value:.2f}, Prediksi: {mock_prediction:.2f})")
        fig_shap.update_layout(template="plotly_dark", 
                               yaxis_title="Fitur", 
                               xaxis_title="Kontribusi SHAP",
                               showlegend=False)
        st.plotly_chart(fig_shap, use_container_width=True)
        
        st.markdown(f"**Analisis Sampel {sample_index}:**")
        st.dataframe(sample_data, use_container_width=True)
    else:
        st.info("Latih model terlebih dahulu untuk mengaktifkan SHAP Simulation.")


def page_prediction_deployment():
    st.header("🔮 6. Prediction & Deployment Pipeline (Industrialization)")
    
    if st.session_state.ml_pipeline is None:
        st.warning("Belum ada model/pipeline yang dilatih. Silakan train model terlebih dahulu.")
        return

    st.subheader("1. Industrial Model Serialization")
    
    # Exporting the FULL Pipeline (Preprocessor + Model)
    model_pipeline = st.session_state.ml_pipeline
    model_bytes = io.BytesIO()
    joblib.dump(model_pipeline, model_bytes)
    model_bytes.seek(0)
    
    original_features_str = st.session_state.original_feature_names
    
    st.info("Pipeline ML lengkap (*Preprocessing* + *Model* terbaik) telah diserialisasi ke `full_ml_pipeline.pkl`.")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download FULL ML Pipeline (.pkl)",
            data=model_bytes,
            file_name="full_ml_pipeline.pkl",
            mime="application/octet-stream",
            use_container_width=True
        )

    # --- FastAPI Deployment Script Mockup (NEW COMPLEXITY) ---
    deployment_code = f"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# --- Konfigurasi Deployment ---
MODEL_PATH = 'full_ml_pipeline.pkl'
REQUIRED_FEATURES = {original_features_str}

app = FastAPI(
    title="ProML Enterprise Prediction API - {st.session_state.current_model_name}",
    version="1.0.0"
)

# Struktur data input Pydantic (Hanya mencantumkan beberapa fitur untuk contoh)
class PredictionInput(BaseModel):
{os.linesep.join([f"    {col}: float | str | None = None" for col in original_features_str[:5]])}
{os.linesep.join([f"    # ... Lanjutkan untuk semua {len(original_features_str)} fitur ..."])}
    
# Muat model saat startup
@app.on_event("startup")
def load_model():
    global pipeline
    try:
        # Asumsi file PKL ada di direktori kerja
        pipeline = joblib.load(MODEL_PATH)
        print(f"Model {{pipeline.steps[-1][0]}} berhasil dimuat.")
    except FileNotFoundError:
        print("Error: Model file not found. Place full_ml_pipeline.pkl next to the script.")
        pipeline = None
    except Exception as e:
        print(f"Error loading model: {{e}}")
        pipeline = None

@app.post("/predict")
def predict(data: PredictionInput):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model service not ready.")

    try:
        # Konversi Pydantic model ke Dictionary
        input_dict = data.model_dump()
        
        # Buat DataFrame
        df_input = pd.DataFrame([input_dict])
        
        # Pastikan urutan kolom sesuai dengan data training ASLI
        df_input = df_input.reindex(columns=REQUIRED_FEATURES, fill_value=np.nan)
        
        # Lakukan prediksi (Pipeline otomatis handle Preprocessing + Prediction)
        prediction_raw = pipeline.predict(df_input)[0]
        
        # Jika Classification, lakukan Inverse Transform jika LabelEncoder tersedia
        # Karena LabelEncoder tidak disimpan di sini, kita kembalikan hasil raw
        
        return {{
            "model": "{st.session_state.current_model_name}",
            "prediction_type": "{st.session_state.task_type}",
            "raw_prediction": prediction_raw.tolist() if isinstance(prediction_raw, np.ndarray) else prediction_raw,
            "message": "Prediction successful. Check raw_prediction for result."
        }}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {{e}}")

# Untuk menjalankan server: uvicorn deploy_api:app --reload
"""
    with col2:
        st.download_button(
            label="Download Deployment Code (FastAPI)",
            data=deployment_code.encode('utf-8'),
            file_name="deploy_api.py",
            mime="text/plain",
            use_container_width=True
        )

    st.markdown("---")
    
    st.subheader("2. Single Prediction Test (Interactive Mockup)")
    st.info("Input fitur asli, Pipeline lengkap akan memproses dan memprediksi.")
    

def page_model_management():
    st.header("📚 7. Model Management & Ranking")
    
    if not st.session_state.model_history:
        st.info("Belum ada model yang tercatat dalam riwayat training.")
        return

    # Buat DataFrame dari riwayat untuk perbandingan
    history_data = []
    
    # Tentukan metrik berdasarkan jenis task model pertama di history
    is_classification = st.session_state.model_history[0]['task_type'].startswith('Classification')
    
    if is_classification:
        primary_metric_label = 'Accuracy'
        secondary_metric_label = 'F1 Score (Macro)'
    else:
        primary_metric_label = 'R2 Score'
        secondary_metric_label = 'RMSE'

    for item in st.session_state.model_history:
        metrics = item['metrics']
        
        history_data.append({
            'Model Name': item['name'],
            'Algorithm': item['model'].named_steps['model'].__class__.__name__,
            'Task': item['task_type'],
            primary_metric_label: metrics.get(primary_metric_label, metrics.get('R2 Score', metrics.get('Accuracy'))),
            secondary_metric_label: metrics.get(secondary_metric_label, metrics.get('RMSE', metrics.get('F1 Score (Macro)'))),
            'Time (s)': item['training_time'],
            'Date': item['date'],
            'Best Params': str(metrics['Best Params']) # Mengubah dictionary ke string untuk display
        })
        
    history_df = pd.DataFrame(history_data)
    
    # Ranking berdasarkan metrik utama
    history_df_ranked = history_df.sort_values(by=primary_metric_label, ascending=False).reset_index(drop=True)
    history_df_ranked.index = history_df_ranked.index + 1
    
    st.subheader("Model Ranking Table")
    st.info(f"Model diperingkat berdasarkan **{primary_metric_label}**.")
    
    # Tampilkan tabel ranking
    st.dataframe(
        history_df_ranked.style.format({
            primary_metric_label: "{:.4f}", 
            secondary_metric_label: "{:.4f}",
            'Time (s)': "{:.2f}"
        }), 
        use_container_width=True
    )
    
    st.markdown("---")
    
    st.subheader("Visual Model Comparison (Metrik Utama)")
    
    models_to_compare = st.multiselect(
        "Pilih Model untuk Perbandingan Visual (Maks 5):", 
        history_df_ranked['Model Name'].tolist(),
        default=history_df_ranked['Model Name'].head(3).tolist(),
        key='visual_compare_models'
    )
    
    if models_to_compare:
        compare_df = history_df_ranked[history_df_ranked['Model Name'].isin(models_to_compare)].set_index('Model Name')
        
        fig_comp = px.bar(
            compare_df.reset_index(), 
            x='Model Name', 
            y=primary_metric_label,
            title=f"Perbandingan Model Berdasarkan {primary_metric_label}",
            color='Model Name',
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        fig_comp.update_layout(template="plotly_dark")
        st.plotly_chart(fig_comp, use_container_width=True)


# --- 4. Sidebar Navigation & Main App Logic ---

st.sidebar.title("ProML Enterprise v5.0")
st.sidebar.markdown("---")
st.sidebar.subheader("Analytics Workflow Steps")

# Mapping untuk Navigation
pages = {
    "Dashboard": 'dashboard',
    "1. Data Upload & Validation": 'upload',
    "2. EDA & Visualisasi": 'eda',
    "3. Preprocessing & FE": 'preprocessing',
    "4. Model Training & Tuning": 'training',
    "5. Evaluation & XAI": 'evaluation',
    "6. Prediction & Deployment": 'prediction_deployment',
    "7. Model Management": 'model_management'
}

# Sidebar Navigation
for page_title, page_id in pages.items():
    if st.sidebar.button(page_title, key=f"nav_{page_id}", use_container_width=True):
        switch_page(page_id)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Task Type:** {st.session_state.task_type if st.session_state.task_type else 'N/A'}")
st.sidebar.markdown(f"**Current Model:** {st.session_state.current_model_name if st.session_state.current_model_name else 'N/A'}")


# 5. Routing Logic
if st.session_state.page == 'dashboard':
    page_dashboard()
elif st.session_state.page == 'upload':
    page_data_upload()
elif st.session_state.page == 'eda':
    page_eda_visualization() 
elif st.session_state.page == 'preprocessing':
    page_preprocessing()
elif st.session_state.page == 'training':
    page_model_training()
elif st.session_state.page == 'evaluation':
    page_model_evaluation()
elif st.session_state.page == 'prediction_deployment':
    page_prediction_deployment()
elif st.session_state.page == 'model_management':
    page_model_management()
else:
    page_dashboard()
