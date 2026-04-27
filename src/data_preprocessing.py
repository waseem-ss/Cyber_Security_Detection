"""
Enhanced Data Preprocessing for Cyber Security Threat Detection.
Handles all CICIDS2017 attack types with class imbalance management.
"""
import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(dataset_dir, sample_per_file=100000, min_class_samples=5):
    """
    Loads ALL CICIDS2017 CSV files with guaranteed representation of every attack class.
    Uses stratified sampling to ensure rare attacks (Heartbleed, Infiltration) are included.
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, label_encoder,
        num_features, feature_names, class_weights_dict
    """
    all_files = glob.glob(os.path.join(dataset_dir, "*.csv"))
    
    df_list = []
    print(f"Loading data from {len(all_files)} files...")
    for file in sorted(all_files):
        try:
            df = pd.read_csv(file, low_memory=False, encoding='utf-8', encoding_errors='replace')
            df.columns = df.columns.str.strip()
            
            # Find label column
            label_col = None
            for col in df.columns:
                if col.strip().lower() == 'label':
                    label_col = col
                    break
            if label_col is None:
                possible = [c for c in df.columns if 'label' in c.lower()]
                if possible:
                    label_col = possible[0]
            if label_col and label_col != 'Label':
                df.rename(columns={label_col: 'Label'}, inplace=True)
            
            # Stratified sampling: keep ALL attack samples, subsample BENIGN
            if 'Label' in df.columns:
                attack_df = df[df['Label'] != 'BENIGN']
                benign_df = df[df['Label'] == 'BENIGN']
                
                benign_sample_size = min(len(benign_df), sample_per_file)
                if len(benign_df) > benign_sample_size:
                    benign_df = benign_df.sample(n=benign_sample_size, random_state=42)
                
                df = pd.concat([benign_df, attack_df], ignore_index=True)
            
            # Sanitize labels: replace unicode chars with ASCII equivalents
            if 'Label' in df.columns:
                df['Label'] = df['Label'].str.encode('ascii', 'replace').str.decode('ascii')
                df['Label'] = df['Label'].str.strip()
            
            df_list.append(df)
            basename = os.path.basename(file)
            attack_types = df['Label'].unique() if 'Label' in df.columns else []
            print(f"  Loaded {len(df):>7} rows from {basename} | Attacks: {list(attack_types)}")
        except Exception as e:
            print(f"  ERROR loading {file}: {e}")
    
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"\nTotal rows loaded: {len(combined_df)}")
    
    # Show class distribution
    print("\n--- Class Distribution ---")
    class_dist = combined_df['Label'].value_counts()
    for cls, count in class_dist.items():
        safe_cls = str(cls).encode('ascii', 'replace').decode('ascii')
        print(f"  {safe_cls:>40s}: {count:>7}")
    
    # Clean data
    combined_df.dropna(axis=1, how='all', inplace=True)
    combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if 'Label' not in combined_df.columns:
        raise ValueError("Could not find 'Label' column in dataset.")
    
    # Filter out classes with too few samples
    class_counts = combined_df['Label'].value_counts()
    valid_classes = class_counts[class_counts >= min_class_samples].index
    removed_classes = class_counts[class_counts < min_class_samples].index.tolist()
    if removed_classes:
        print(f"\nWARNING: Removing classes with <{min_class_samples} samples: {removed_classes}")
    combined_df = combined_df[combined_df['Label'].isin(valid_classes)]
    
    # Separate features and target
    y = combined_df['Label']
    X = combined_df.drop('Label', axis=1)
    
    # Drop non-numeric columns
    cols_to_drop = [col for col in X.columns if X[col].dtype == 'object']
    if cols_to_drop:
        print(f"Dropping non-numeric columns: {cols_to_drop}")
        X.drop(cols_to_drop, axis=1, inplace=True)
    
    feature_names = list(X.columns)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Label Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"\nTotal classes: {len(le.classes_)}")
    print(f"Classes: {list(le.classes_)}")
    
    # Compute class weights for imbalance handling
    class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
    class_weights_dict = dict(zip(np.unique(y_encoded), class_weights))
    print(f"\nClass weights computed for {len(class_weights_dict)} classes")
    
    # Train/Val/Test Split (70/15/15) with stratification
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Reshape for CNN-LSTM: (samples, timesteps, features)
    # Use timesteps=1 per flow, but the architecture handles it properly
    X_train_seq = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val_seq = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    X_test_seq = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    print(f"\nX_train: {X_train_seq.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val_seq.shape}, y_val:   {y_val.shape}")
    print(f"X_test:  {X_test_seq.shape}, y_test:  {y_test.shape}")
    
    return (X_train_seq, X_val_seq, X_test_seq,
            y_train, y_val, y_test,
            le, X_train.shape[1], feature_names, class_weights_dict)
