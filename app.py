# app.py - Streamlit ML Application
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_squared_error, r2_score, mean_absolute_error,
                             silhouette_score)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, chi2, mutual_info_classif
import pickle
import io

# Page configuration
st.set_page_config(
    page_title="Aplikasi ML dengan Scikit-Learn",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    h1 {
        color: #667eea;
        padding-bottom: 1rem;
    }
    h2 {
        color: #764ba2;
        padding-top: 1rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
    }
    .stSelectbox, .stMultiSelect {
        padding: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'encoders' not in st.session_state:
    st.session_state.encoders = {}
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

# Header
st.title("🤖 Aplikasi Machine Learning dengan Scikit-Learn")
st.markdown("**Platform Lengkap untuk Analisis Data, Preprocessing, dan Pemodelan ML**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📋 Menu Navigasi")
    page = st.radio(
        "Pilih Halaman:",
        ["📤 Upload Data", "🔍 Eksplorasi Data", "⚙️ Preprocessing", 
         "🎯 Feature Selection", "📊 Visualisasi", "🧠 Model ML", "🔮 Prediksi"]
    )
    
    st.markdown("---")
    st.info("💡 **Tips**: Upload dataset CSV terlebih dahulu untuk memulai!")
    
    if st.session_state.df is not None:
        st.success(f"✅ Data loaded: {st.session_state.df.shape[0]} rows, {st.session_state.df.shape[1]} columns")

# ===== PAGE 1: UPLOAD DATA =====
if page == "📤 Upload Data":
    st.header("📤 Upload Dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Pilih file CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.session_state.df_processed = df.copy()
                
                st.success(f"✅ Data berhasil diupload!")
                
                # Auto-detect data types
                with st.expander("🔍 Deteksi Otomatis Tipe Data", expanded=True):
                    # Check for datetime columns
                    potential_datetime = []
                    for col in df.select_dtypes(include=['object']).columns:
                        try:
                            pd.to_datetime(df[col])
                            potential_datetime.append(col)
                        except:
                            pass
                    
                    if potential_datetime:
                        st.warning(f"⚠️ Kolom datetime terdeteksi: {', '.join(potential_datetime)}")
                        st.info("💡 Silakan ke halaman Preprocessing untuk mengkonversi kolom datetime")
                    
                    # Check for categorical columns
                    categorical = df.select_dtypes(include=['object']).columns.tolist()
                    if categorical:
                        st.warning(f"⚠️ Kolom kategorikal terdeteksi: {', '.join(categorical)}")
                        st.info("💡 Silakan ke halaman Preprocessing untuk encoding kolom kategorikal")
                
                # Display basic info
                st.subheader("Informasi Dataset")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Jumlah Baris", df.shape[0])
                col_b.metric("Jumlah Kolom", df.shape[1])
                col_c.metric("Missing Values", df.isnull().sum().sum())
                
                # Preview data
                st.subheader("Preview Data")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Column types
                st.subheader("Tipe Data Kolom")
                dtype_df = pd.DataFrame({
                    'Kolom': df.columns,
                    'Tipe Data': df.dtypes.values,
                    'Non-Null Count': df.count().values,
                    'Null Count': df.isnull().sum().values
                })
                st.dataframe(dtype_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error membaca file: {str(e)}")
    
    with col2:
        st.info("""
        **Format Data:**
        - Format: CSV
        - Header: Ya
        - Encoding: UTF-8
        
        **Contoh Dataset:**
        - Iris Dataset
        - Titanic Dataset
        - Housing Prices
        - dll.
        """)

# ===== PAGE 2: EKSPLORASI DATA =====
elif page == "🔍 Eksplorasi Data":
    st.header("🔍 Eksplorasi Data")
    
    if st.session_state.df is None:
        st.warning("⚠️ Silakan upload data terlebih dahulu!")
    else:
        df = st.session_state.df
        
        tab1, tab2, tab3 = st.tabs(["📊 Statistik Deskriptif", "📈 Info Detail", "🔢 Data Numerik"])
        
        with tab1:
            st.subheader("Statistik Deskriptif")
            st.dataframe(df.describe(), use_container_width=True)
            
            # Missing values analysis
            st.subheader("Analisis Missing Values")
            missing = df.isnull().sum()
            missing_percent = 100 * missing / len(df)
            missing_df = pd.DataFrame({
                'Kolom': missing.index,
                'Missing Count': missing.values,
                'Percentage': missing_percent.values
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
            
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.barh(missing_df['Kolom'], missing_df['Percentage'], color='#667eea')
                ax.set_xlabel('Percentage of Missing Values')
                ax.set_title('Missing Values per Column')
                st.pyplot(fig)
            else:
                st.success("✅ Tidak ada missing values!")
        
        with tab2:
            st.subheader("Informasi Detail Dataset")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
            
            # Value counts for categorical columns
            st.subheader("Distribusi Kolom Kategorikal")
            cat_cols = df.select_dtypes(include=['object']).columns
            
            if len(cat_cols) > 0:
                selected_cat = st.selectbox("Pilih kolom kategorikal:", cat_cols)
                value_counts = df[selected_cat].value_counts()
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.dataframe(value_counts, use_container_width=True)
                with col2:
                    fig, ax = plt.subplots()
                    value_counts.head(10).plot(kind='bar', ax=ax, color='#667eea')
                    ax.set_title(f'Distribusi {selected_cat}')
                    ax.set_xlabel(selected_cat)
                    ax.set_ylabel('Count')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
            else:
                st.info("Tidak ada kolom kategorikal")
        
        with tab3:
            st.subheader("Statistik Kolom Numerik")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                selected_num = st.selectbox("Pilih kolom numerik:", numeric_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Statistik:**")
                    stats = df[selected_num].describe()
                    st.dataframe(stats)
                
                with col2:
                    st.write("**Distribusi:**")
                    fig, ax = plt.subplots()
                    df[selected_num].hist(bins=30, ax=ax, color='#667eea', edgecolor='black')
                    ax.set_title(f'Histogram {selected_num}')
                    ax.set_xlabel(selected_num)
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)

# ===== PAGE 3: PREPROCESSING =====
elif page == "⚙️ Preprocessing":
    st.header("⚙️ Preprocessing Data")
    
    if st.session_state.df is None:
        st.warning("⚠️ Silakan upload data terlebih dahulu!")
    else:
        df = st.session_state.df_processed.copy()
        
        # Quick Auto-Preprocessing
        st.subheader("🚀 Quick Auto-Preprocessing")
        st.info("Otomatis handle missing values, encode categorical, dan convert datetime")
        
        if st.button("⚡ Auto-Preprocess (Recommended)", type="primary", key="auto_preprocess"):
            with st.spinner("Memproses data..."):
                # 1. Handle datetime columns
                datetime_cols = []
                for col in df.select_dtypes(include=['object']).columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        datetime_cols.append(col)
                    except:
                        pass
                
                if datetime_cols:
                    for col in datetime_cols:
                        df[f'{col}_year'] = df[col].dt.year
                        df[f'{col}_month'] = df[col].dt.month
                        df[f'{col}_day'] = df[col].dt.day
                        df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                        df = df.drop(col, axis=1)
                    st.success(f"✅ Converted {len(datetime_cols)} datetime columns")
                
                # 2. Encode categorical columns
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                if categorical_cols:
                    for col in categorical_cols:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                        st.session_state.encoders[col] = le
                    st.success(f"✅ Encoded {len(categorical_cols)} categorical columns")
                
                # 3. Handle missing values
                missing_count = df.isnull().sum().sum()
                if missing_count > 0:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        imputer = SimpleImputer(strategy='mean')
                        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                    st.success(f"✅ Handled {missing_count} missing values")
                
                st.session_state.df_processed = df
                st.success("🎉 Auto-preprocessing selesai! Data siap untuk modeling")
        
        st.markdown("---")
        st.subheader("🎛️ Manual Preprocessing (Advanced)")
        
        # Handle Missing Values
        st.subheader("1️⃣ Handle Missing Values")
        col1, col2 = st.columns(2)
        
        with col1:
            missing_strategy = st.selectbox(
                "Strategi untuk missing values:",
                ["Tidak ada aksi", "Mean (numerik)", "Median (numerik)", "Most Frequent", "Drop rows"]
            )
        
        with col2:
            if st.button("Terapkan Missing Values Handler", key="missing"):
                if missing_strategy != "Tidak ada aksi":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    categorical_cols = df.select_dtypes(include=['object']).columns
                    
                    if missing_strategy == "Drop rows":
                        df = df.dropna()
                        st.success(f"✅ Rows dengan missing values dihapus. Shape sekarang: {df.shape}")
                    else:
                        strategy_map = {
                            "Mean (numerik)": "mean",
                            "Median (numerik)": "median",
                            "Most Frequent": "most_frequent"
                        }
                        strategy = strategy_map[missing_strategy]
                        
                        if len(numeric_cols) > 0:
                            imputer = SimpleImputer(strategy=strategy if strategy != "most_frequent" else "mean")
                            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                        
                        if len(categorical_cols) > 0 and strategy == "most_frequent":
                            imputer = SimpleImputer(strategy="most_frequent")
                            df[categorical_cols] = imputer.fit_transform(df[categorical_cols])
                        
                        st.success(f"✅ Missing values berhasil dihandle dengan strategi: {missing_strategy}")
                    
                    st.session_state.df_processed = df
        
        st.markdown("---")
        
        # Encoding
        st.subheader("2️⃣ Encoding Kolom Kategorikal & Datetime")
        
        # Detect categorical and datetime columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = []
        
        # Check for datetime-like columns
        for col in categorical_cols[:]:
            try:
                pd.to_datetime(df[col])
                datetime_cols.append(col)
                categorical_cols.remove(col)
            except:
                pass
        
        # Handle datetime columns
        if len(datetime_cols) > 0:
            st.write("**Kolom Datetime Terdeteksi:**")
            datetime_to_process = st.multiselect(
                "Pilih kolom datetime untuk dikonversi:",
                datetime_cols,
                default=datetime_cols
            )
            
            if st.button("Konversi Datetime", key="datetime_convert"):
                if datetime_to_process:
                    for col in datetime_to_process:
                        df[col] = pd.to_datetime(df[col])
                        # Extract features
                        df[f'{col}_year'] = df[col].dt.year
                        df[f'{col}_month'] = df[col].dt.month
                        df[f'{col}_day'] = df[col].dt.day
                        df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                        # Drop original column
                        df = df.drop(col, axis=1)
                    
                    st.success(f"✅ {len(datetime_to_process)} kolom datetime berhasil dikonversi!")
                    st.session_state.df_processed = df
            
            st.markdown("---")
        
        # Handle categorical columns
        if len(categorical_cols) > 0:
            st.write("**Kolom Kategorikal:**")
            col1, col2 = st.columns(2)
            
            with col1:
                cols_to_encode = st.multiselect(
                    "Pilih kolom untuk di-encode:",
                    categorical_cols,
                    default=categorical_cols
                )
            
            with col2:
                encoding_method = st.selectbox(
                    "Metode encoding:",
                    ["Label Encoding", "One-Hot Encoding"]
                )
            
            if st.button("Terapkan Encoding", key="encoding"):
                if cols_to_encode:
                    for col in cols_to_encode:
                        if encoding_method == "Label Encoding":
                            le = LabelEncoder()
                            df[col] = le.fit_transform(df[col].astype(str))
                            st.session_state.encoders[col] = le
                        else:  # One-Hot Encoding
                            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                            df = pd.concat([df, dummies], axis=1)
                            df = df.drop(col, axis=1)
                    
                    st.success(f"✅ {len(cols_to_encode)} kolom berhasil di-encode!")
                    st.session_state.df_processed = df
                else:
                    st.warning("Pilih minimal satu kolom untuk di-encode")
        else:
            st.info("Tidak ada kolom kategorikal untuk di-encode")
        
        st.markdown("---")
        
        # Scaling
        st.subheader("3️⃣ Scaling Data")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                scaling_method = st.selectbox(
                    "Metode scaling:",
                    ["Tidak ada scaling", "Standard Scaler (Z-score)", "Min-Max Scaler (0-1)"]
                )
            
            with col2:
                cols_to_scale = st.multiselect(
                    "Pilih kolom untuk di-scale:",
                    numeric_cols,
                    default=numeric_cols
                )
            
            if st.button("Terapkan Scaling", key="scaling"):
                if scaling_method != "Tidak ada scaling" and cols_to_scale:
                    if scaling_method == "Standard Scaler (Z-score)":
                        scaler = StandardScaler()
                    else:
                        scaler = MinMaxScaler()
                    
                    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
                    st.session_state.scaler = scaler
                    st.session_state.df_processed = df
                    st.success(f"✅ Scaling berhasil diterapkan dengan {scaling_method}")
        else:
            st.info("Tidak ada kolom numerik untuk di-scale")
        
        st.markdown("---")
        
        # Preview processed data
        st.subheader("Preview Data Setelah Preprocessing")
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah Baris", df.shape[0])
        col2.metric("Jumlah Kolom", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

# ===== PAGE 4: FEATURE SELECTION =====
elif page == "🎯 Feature Selection":
    st.header("🎯 Feature Selection")
    
    if st.session_state.df_processed is None:
        st.warning("⚠️ Silakan lakukan preprocessing terlebih dahulu!")
    else:
        df = st.session_state.df_processed
        
        st.subheader("1️⃣ Pilih Target dan Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_col = st.selectbox(
                "Target Column (Label):",
                [""] + df.columns.tolist()
            )
        
        with col2:
            if target_col:
                available_features = [col for col in df.columns if col != target_col]
                selected_features = st.multiselect(
                    "Pilih Features:",
                    available_features,
                    default=available_features
                )
        
        if target_col and selected_features:
            st.markdown("---")
            st.subheader("2️⃣ Feature Selection Otomatis (Opsional)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selection_method = st.selectbox(
                    "Metode Selection:",
                    ["SelectKBest", "Manual (skip)"]
                )
            
            with col2:
                if selection_method == "SelectKBest":
                    k_best = st.number_input(
                        "Jumlah fitur terbaik (k):",
                        min_value=1,
                        max_value=len(selected_features),
                        value=min(5, len(selected_features))
                    )
            
            with col3:
                if selection_method == "SelectKBest":
                    score_func = st.selectbox(
                        "Score Function:",
                        ["f_classif", "f_regression", "mutual_info_classif"]
                    )
            
            if st.button("Analisis Features", key="feature_select"):
                X = df[selected_features]
                y = df[target_col]
                
                if selection_method == "SelectKBest":
                    score_func_map = {
                        "f_classif": f_classif,
                        "f_regression": f_regression,
                        "mutual_info_classif": mutual_info_classif
                    }
                    
                    selector = SelectKBest(score_func=score_func_map[score_func], k=k_best)
                    selector.fit(X, y)
                    
                    scores = pd.DataFrame({
                        'Feature': selected_features,
                        'Score': selector.scores_
                    }).sort_values('Score', ascending=False)
                    
                    st.subheader("Feature Scores")
                    st.dataframe(scores, use_container_width=True)
                    
                    # Plot scores
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(scores['Feature'], scores['Score'], color='#667eea')
                    ax.set_xlabel('Score')
                    ax.set_title(f'Feature Importance ({score_func})')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    best_features = scores.head(k_best)['Feature'].tolist()
                    st.success(f"✅ Top {k_best} features: {', '.join(best_features)}")
                
                else:
                    st.info(f"✅ Menggunakan {len(selected_features)} features yang dipilih secara manual")
                
                # Store for model training
                st.session_state.selected_features = selected_features
                st.session_state.target_col = target_col

# ===== PAGE 5: VISUALISASI =====
elif page == "📊 Visualisasi":
    st.header("📊 Visualisasi Data")
    
    if st.session_state.df_processed is None:
        st.warning("⚠️ Silakan upload dan proses data terlebih dahulu!")
    else:
        df = st.session_state.df_processed
        
        viz_type = st.selectbox(
            "Pilih Jenis Visualisasi:",
            ["Correlation Heatmap", "Distribution Plot", "Box Plot", "Scatter Plot", "Pairplot"]
        )
        
        if viz_type == "Correlation Heatmap":
            st.subheader("📈 Correlation Heatmap")
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) > 1:
                fig, ax = plt.subplots(figsize=(12, 8))
                corr = numeric_df.corr()
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
                ax.set_title('Correlation Matrix')
                st.pyplot(fig)
            else:
                st.warning("Tidak cukup kolom numerik untuk correlation heatmap")
        
        elif viz_type == "Distribution Plot":
            st.subheader("📊 Distribution Plot")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_col = st.selectbox("Pilih kolom:", numeric_cols)
                
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # Histogram
                axes[0].hist(df[selected_col].dropna(), bins=30, color='#667eea', edgecolor='black')
                axes[0].set_title(f'Histogram - {selected_col}')
                axes[0].set_xlabel(selected_col)
                axes[0].set_ylabel('Frequency')
                
                # Box plot
                axes[1].boxplot(df[selected_col].dropna())
                axes[1].set_title(f'Box Plot - {selected_col}')
                axes[1].set_ylabel(selected_col)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        elif viz_type == "Box Plot":
            st.subheader("📦 Box Plot")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_cols = st.multiselect("Pilih kolom:", numeric_cols, default=numeric_cols[:5])
                
                if selected_cols:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    df[selected_cols].boxplot(ax=ax)
                    ax.set_title('Box Plot Comparison')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
        
        elif viz_type == "Scatter Plot":
            st.subheader("🔵 Scatter Plot")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis:", numeric_cols)
                with col2:
                    y_col = st.selectbox("Y-axis:", [c for c in numeric_cols if c != x_col])
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(df[x_col], df[y_col], alpha=0.6, color='#667eea')
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f'{x_col} vs {y_col}')
                st.pyplot(fig)
        
        elif viz_type == "Pairplot":
            st.subheader("🎯 Pairplot")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                selected_cols = st.multiselect(
                    "Pilih kolom (max 5 untuk performa):",
                    numeric_cols,
                    default=numeric_cols[:min(5, len(numeric_cols))]
                )
                
                if len(selected_cols) >= 2:
                    if st.button("Generate Pairplot"):
                        with st.spinner("Generating pairplot..."):
                            fig = sns.pairplot(df[selected_cols])
                            st.pyplot(fig)

# ===== PAGE 6: MODEL ML =====
elif page == "🧠 Model ML":
    st.header("🧠 Machine Learning Model")
    
    if st.session_state.df_processed is None:
        st.warning("⚠️ Silakan lakukan preprocessing terlebih dahulu!")
    elif 'selected_features' not in st.session_state or 'target_col' not in st.session_state:
        st.warning("⚠️ Silakan lakukan feature selection terlebih dahulu!")
    else:
        df = st.session_state.df_processed
        target_col = st.session_state.target_col
        selected_features = st.session_state.selected_features
        
        st.subheader("1️⃣ Pilih Model dan Algoritma")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Tipe Model:",
                ["Classification", "Regression", "Clustering"]
            )
        
        with col2:
            if model_type == "Classification":
                algorithm = st.selectbox(
                    "Algoritma:",
                    ["Random Forest", "Logistic Regression", "Decision Tree", 
                     "SVM", "Naive Bayes", "KNN", "Gradient Boosting"]
                )
            elif model_type == "Regression":
                algorithm = st.selectbox(
                    "Algoritma:",
                    ["Random Forest", "Linear Regression", "Decision Tree",
                     "SVR", "KNN", "Ridge", "Lasso", "Gradient Boosting"]
                )
            else:  # Clustering
                algorithm = st.selectbox(
                    "Algoritma:",
                    ["K-Means", "DBSCAN", "Agglomerative"]
                )
        
        st.markdown("---")
        st.subheader("2️⃣ Parameter Training")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if model_type != "Clustering":
                test_size = st.slider("Test Size (%):", 10, 50, 20) / 100
            else:
                if algorithm == "K-Means":
                    n_clusters = st.number_input("Number of Clusters:", 2, 10, 3)
                elif algorithm == "DBSCAN":
                    eps = st.number_input("Epsilon:", 0.1, 10.0, 0.5)
                    min_samples = st.number_input("Min Samples:", 1, 20, 5)
                else:  # Agglomerative
                    n_clusters = st.number_input("Number of Clusters:", 2, 10, 3)
        
        with col2:
            random_state = st.number_input("Random State:", 0, 100, 42)
        
        with col3:
            if model_type != "Clustering":
                use_cv = st.checkbox("Cross Validation (5-fold)")
        
        st.markdown("---")
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model..."):
                try:
                    X = df[selected_features].copy()
                    
                    # Check for non-numeric columns
                    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
                    
                    if len(non_numeric_cols) > 0:
                        st.error(f"❌ Kolom berikut masih non-numerik: {', '.join(non_numeric_cols)}")
                        st.warning("⚠️ Silakan lakukan encoding pada kolom kategorikal di halaman Preprocessing terlebih dahulu!")
                        st.stop()
                    
                    # Check for missing values
                    if X.isnull().sum().sum() > 0:
                        st.warning("⚠️ Data masih memiliki missing values. Menggunakan mean imputation...")
                        imputer = SimpleImputer(strategy='mean')
                        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
                    
                    if model_type != "Clustering":
                        y = df[target_col].copy()
                        
                        # Check if target is categorical string
                        if y.dtype == 'object':
                            st.info("🔄 Target kolom adalah kategorikal, melakukan label encoding...")
                            le_target = LabelEncoder()
                            y = le_target.fit_transform(y)
                            st.session_state.target_encoder = le_target
                        
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=random_state
                        )
                        
                        # Store for prediction
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                    
                    # Initialize model
                    if model_type == "Classification":
                        models = {
                            "Random Forest": RandomForestClassifier(random_state=random_state),
                            "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000),
                            "Decision Tree": DecisionTreeClassifier(random_state=random_state),
                            "SVM": SVC(random_state=random_state),
                            "Naive Bayes": GaussianNB(),
                            "KNN": KNeighborsClassifier(),
                            "Gradient Boosting": GradientBoostingClassifier(random_state=random_state)
                        }
                    else:  # Regression
                        models = {
                            "Random Forest": RandomForestRegressor(random_state=random_state),
                            "Linear Regression": LinearRegression(),
                            "Decision Tree": DecisionTreeRegressor(random_state=random_state),
                            "SVR": SVR(),
                            "KNN": KNeighborsRegressor(),
                            "Ridge": Ridge(random_state=random_state),
                            "Lasso": Lasso(random_state=random_state),
                            "Gradient Boosting": GradientBoostingRegressor(random_state=random_state)
                        }
                    
                    model = models[algorithm]
                    model.fit(X_train, y_train)
                    st.session_state.model = model
                    
                    # Predictions
                    y_pred = model.predict(X_test)
                    
                    # Metrics
                    st.success("✅ Model berhasil ditraining!")
                    
                    st.subheader("📊 Hasil Training")
                    
                    if model_type == "Classification":
                        train_acc = accuracy_score(y_train, model.predict(X_train))
                        test_acc = accuracy_score(y_test, y_pred)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Train Accuracy", f"{train_acc:.4f}")
                        col2.metric("Test Accuracy", f"{test_acc:.4f}")
                        
                        if use_cv:
                            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                            col3.metric("CV Score (mean)", f"{cv_scores.mean():.4f}")
                        
                        # Classification Report
                        st.subheader("Classification Report")
                        report = classification_report(y_test, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df, use_container_width=True)
                        
                        # Confusion Matrix
                        st.subheader("Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_title('Confusion Matrix')
                        st.pyplot(fig)
                    
                    else:  # Regression
                        train_pred = model.predict(X_train)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        train_r2 = r2_score(y_train, train_pred)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("RMSE", f"{rmse:.4f}")
                        col2.metric("MAE", f"{mae:.4f}")
                        col3.metric("R² Score", f"{r2:.4f}")
                        col4.metric("Train R²", f"{train_r2:.4f}")
                        
                        if use_cv:
                            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                            st.metric("CV R² Score (mean)", f"{cv_scores.mean():.4f}")
                        
                        # Prediction vs Actual Plot
                        st.subheader("Prediction vs Actual")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.scatter(y_test, y_pred, alpha=0.6, color='#667eea')
                        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                               'r--', lw=2, label='Perfect Prediction')
                        ax.set_xlabel('Actual Values')
                        ax.set_ylabel('Predicted Values')
                        ax.set_title('Actual vs Predicted')
                        ax.legend()
                        st.pyplot(fig)
                        
                        # Residual Plot
                        st.subheader("Residual Plot")
                        residuals = y_test - y_pred
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.scatter(y_pred, residuals, alpha=0.6, color='#667eea')
                        ax.axhline(y=0, color='r', linestyle='--', lw=2)
                        ax.set_xlabel('Predicted Values')
                        ax.set_ylabel('Residuals')
                        ax.set_title('Residual Plot')
                        st.pyplot(fig)
                    
                    # Feature Importance (if available)
                    if hasattr(model, 'feature_importances_'):
                        st.subheader("Feature Importance")
                        importance_df = pd.DataFrame({
                            'Feature': selected_features,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(importance_df['Feature'], importance_df['Importance'], color='#667eea')
                        ax.set_xlabel('Importance')
                        ax.set_title('Feature Importance')
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"❌ Error saat training model: {str(e)}")
                    st.error("💡 Pastikan semua kolom sudah numerik dan tidak ada missing values")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Clustering section (outside the classification/regression block)
        if st.button("🚀 Train Clustering Model", type="primary", key="train_cluster") and model_type == "Clustering":
            with st.spinner("Training clustering model..."):
                try:
                    X = df[selected_features].copy()
                    
                    # Check for non-numeric columns
                    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
                    
                    if len(non_numeric_cols) > 0:
                        st.error(f"❌ Kolom berikut masih non-numerik: {', '.join(non_numeric_cols)}")
                        st.warning("⚠️ Silakan lakukan encoding pada kolom kategorikal di halaman Preprocessing terlebih dahulu!")
                        st.stop()
                    
                    # Check for missing values
                    if X.isnull().sum().sum() > 0:
                        st.warning("⚠️ Data masih memiliki missing values. Menggunakan mean imputation...")
                        imputer = SimpleImputer(strategy='mean')
                        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
                    models = {
                        "K-Means": KMeans(n_clusters=n_clusters, random_state=random_state),
                        "DBSCAN": DBSCAN(eps=eps, min_samples=min_samples),
                        "Agglomerative": AgglomerativeClustering(n_clusters=n_clusters)
                    }
                    
                    model = models[algorithm]
                    clusters = model.fit_predict(X)
                    st.session_state.model = model
                    
                    st.success("✅ Clustering berhasil!")
                    
                    # Add cluster labels to dataframe
                    df['Cluster'] = clusters
                    
                    # Metrics
                    st.subheader("📊 Hasil Clustering")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Jumlah Cluster", len(np.unique(clusters)))
                    
                    if len(np.unique(clusters)) > 1:
                        silhouette = silhouette_score(X, clusters)
                        col2.metric("Silhouette Score", f"{silhouette:.4f}")
                    
                    # Cluster distribution
                    st.subheader("Distribusi Cluster")
                    cluster_counts = pd.Series(clusters).value_counts().sort_index()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(cluster_counts, use_container_width=True)
                    with col2:
                        fig, ax = plt.subplots()
                        cluster_counts.plot(kind='bar', ax=ax, color='#667eea')
                        ax.set_xlabel('Cluster')
                        ax.set_ylabel('Count')
                        ax.set_title('Cluster Distribution')
                        st.pyplot(fig)
                    
                    # Visualization (2D projection if more than 2 features)
                    if len(selected_features) >= 2:
                        st.subheader("Visualisasi Cluster")
                        
                        if len(selected_features) == 2:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], 
                                               c=clusters, cmap='viridis', alpha=0.6)
                            ax.set_xlabel(selected_features[0])
                            ax.set_ylabel(selected_features[1])
                            ax.set_title('Cluster Visualization')
                            plt.colorbar(scatter, ax=ax, label='Cluster')
                            st.pyplot(fig)
                        else:
                            # PCA for dimensionality reduction
                            from sklearn.decomposition import PCA
                            pca = PCA(n_components=2)
                            X_pca = pca.fit_transform(X)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                                               c=clusters, cmap='viridis', alpha=0.6)
                            ax.set_xlabel('First Principal Component')
                            ax.set_ylabel('Second Principal Component')
                            ax.set_title('Cluster Visualization (PCA)')
                            plt.colorbar(scatter, ax=ax, label='Cluster')
                            st.pyplot(fig)
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {str(e)}")
        
        st.markdown("---")
        
        # Download Model
        if st.session_state.model is not None:
            st.subheader("💾 Download Model")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Save model
                model_bytes = pickle.dumps(st.session_state.model)
                st.download_button(
                    label="📥 Download Model (.pkl)",
                    data=model_bytes,
                    file_name=f"{algorithm.replace(' ', '_')}_model.pkl",
                    mime="application/octet-stream"
                )
            
            with col2:
                if st.session_state.scaler is not None:
                    scaler_bytes = pickle.dumps(st.session_state.scaler)
                    st.download_button(
                        label="📥 Download Scaler (.pkl)",
                        data=scaler_bytes,
                        file_name="scaler.pkl",
                        mime="application/octet-stream"
                    )

# ===== PAGE 7: PREDIKSI =====
elif page == "🔮 Prediksi":
    st.header("🔮 Prediksi dengan Model")
    
    if st.session_state.model is None:
        st.warning("⚠️ Silakan train model terlebih dahulu!")
    else:
        st.success("✅ Model siap untuk prediksi!")
        
        st.subheader("Input Data untuk Prediksi")
        
        # Get feature names
        if hasattr(st.session_state, 'selected_features'):
            features = st.session_state.selected_features
        else:
            st.error("Feature information tidak tersedia")
            st.stop()
        
        # Method selection
        input_method = st.radio(
            "Metode Input:",
            ["Manual Input", "Upload CSV"]
        )
        
        if input_method == "Manual Input":
            st.write("**Masukkan nilai untuk setiap fitur:**")
            
            # Create input fields
            input_data = {}
            cols = st.columns(2)
            
            for idx, feature in enumerate(features):
                with cols[idx % 2]:
                    input_data[feature] = st.number_input(
                        f"{feature}:",
                        value=0.0,
                        format="%.4f",
                        key=f"input_{feature}"
                    )
            
            if st.button("🔮 Prediksi", type="primary"):
                # Prepare input
                input_df = pd.DataFrame([input_data])
                
                # Apply scaling if available
                if st.session_state.scaler is not None:
                    try:
                        input_df = pd.DataFrame(
                            st.session_state.scaler.transform(input_df),
                            columns=input_df.columns
                        )
                    except:
                        st.warning("Scaler tidak dapat diterapkan pada input ini")
                
                # Make prediction
                prediction = st.session_state.model.predict(input_df)
                
                # Display result
                st.markdown("---")
                st.subheader("📊 Hasil Prediksi")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Input Data:**")
                    st.dataframe(pd.DataFrame([input_data]), use_container_width=True)
                
                with col2:
                    st.write("**Prediksi:**")
                    st.metric("Hasil Prediksi", f"{prediction[0]}")
                
                # Show probability for classification
                if hasattr(st.session_state.model, 'predict_proba'):
                    proba = st.session_state.model.predict_proba(input_df)
                    st.write("**Probabilitas:**")
                    
                    proba_df = pd.DataFrame(
                        proba,
                        columns=[f"Class {i}" for i in range(proba.shape[1])]
                    )
                    st.dataframe(proba_df, use_container_width=True)
                    
                    # Probability chart
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.bar(proba_df.columns, proba[0], color='#667eea')
                    ax.set_ylabel('Probability')
                    ax.set_title('Prediction Probability')
                    ax.set_ylim(0, 1)
                    st.pyplot(fig)
        
        else:  # Upload CSV
            st.write("**Upload file CSV untuk batch prediction:**")
            
            pred_file = st.file_uploader("Pilih file CSV", type=['csv'], key="predict_file")
            
            if pred_file is not None:
                try:
                    pred_df = pd.read_csv(pred_file)
                    st.write("Preview Data:")
                    st.dataframe(pred_df.head(), use_container_width=True)
                    
                    # Check if all features are present
                    missing_features = [f for f in features if f not in pred_df.columns]
                    
                    if missing_features:
                        st.error(f"❌ Missing features: {', '.join(missing_features)}")
                    else:
                        if st.button("🔮 Batch Prediksi", type="primary"):
                            # Prepare data
                            X_pred = pred_df[features]
                            
                            # Apply scaling if available
                            if st.session_state.scaler is not None:
                                try:
                                    X_pred = pd.DataFrame(
                                        st.session_state.scaler.transform(X_pred),
                                        columns=X_pred.columns
                                    )
                                except:
                                    st.warning("Scaler tidak dapat diterapkan")
                            
                            # Make predictions
                            predictions = st.session_state.model.predict(X_pred)
                            
                            # Add predictions to dataframe
                            result_df = pred_df.copy()
                            result_df['Prediction'] = predictions
                            
                            # Add probabilities if available
                            if hasattr(st.session_state.model, 'predict_proba'):
                                proba = st.session_state.model.predict_proba(X_pred)
                                for i in range(proba.shape[1]):
                                    result_df[f'Probability_Class_{i}'] = proba[:, i]
                            
                            st.success(f"✅ Prediksi selesai untuk {len(result_df)} baris!")
                            
                            # Display results
                            st.subheader("📊 Hasil Prediksi")
                            st.dataframe(result_df, use_container_width=True)
                            
                            # Download results
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Hasil Prediksi (CSV)",
                                data=csv,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )
                            
                            # Summary statistics
                            st.subheader("Ringkasan Prediksi")
                            
                            if result_df['Prediction'].dtype in ['int64', 'float64']:
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Mean", f"{result_df['Prediction'].mean():.4f}")
                                col2.metric("Median", f"{result_df['Prediction'].median():.4f}")
                                col3.metric("Min", f"{result_df['Prediction'].min():.4f}")
                                col4.metric("Max", f"{result_df['Prediction'].max():.4f}")
                            
                            # Distribution plot
                            fig, ax = plt.subplots(figsize=(10, 6))
                            result_df['Prediction'].hist(bins=30, ax=ax, color='#667eea', edgecolor='black')
                            ax.set_xlabel('Prediction')
                            ax.set_ylabel('Frequency')
                            ax.set_title('Distribution of Predictions')
                            st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #6c757d; padding: 20px;'>
        <p>🤖 <strong>Aplikasi ML dengan Scikit-Learn</strong></p>
        <p>Built with Streamlit | Powered by Scikit-Learn</p>
    </div>
    """, unsafe_allow_html=True)