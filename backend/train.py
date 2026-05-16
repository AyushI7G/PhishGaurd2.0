import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib

from feature_extraction import (
    extract_email_text_features,
    extract_urls,
    extract_url_features,
    extract_email_address_features,
)
from settings import DATASET_PATHS, MODEL_PATHS


# =====================================================
# HELPERS
# =====================================================


def _read_csv_safe(path: str, *, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        low_memory=False,
        nrows=nrows,
        encoding='utf-8',
        on_bad_lines='skip',
    )



def _clean_binary_labels(series: pd.Series, mapping: dict[str, int]) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.map(mapping).fillna(0).astype(int)



def safe_feature_vector(arr, expected_size):
    """
    Ensures feature vector is always same size.
    Pads/truncates instead of failing.
    """

    arr = np.array(arr, dtype=float).flatten()

    if len(arr) < expected_size:
        padding = np.zeros(expected_size - len(arr))
        arr = np.concatenate([arr, padding])

    elif len(arr) > expected_size:
        arr = arr[:expected_size]

    return arr


# =====================================================
# LOAD DATASETS
# =====================================================

print("Loading datasets...")

email_df = _read_csv_safe(DATASET_PATHS['email'])
url_df = _read_csv_safe(DATASET_PATHS['url'])

print(f"Email dataset rows: {len(email_df)}")
print(f"URL dataset rows: {len(url_df)}")


# =====================================================
# EMAIL DATASET COLUMN DETECTION
# =====================================================

email_col = next(
    (
        c
        for c in email_df.columns
        if any(k in c.lower() for k in ['body', 'text', 'message', 'content'])
    ),
    None,
)

if email_col is None:
    raise Exception(
        f"Could not detect email text column. Found columns: {email_df.columns.tolist()}"
    )

label_col = next(
    (
        c
        for c in email_df.columns
        if 'label' in c.lower() or 'class' in c.lower()
    ),
    None,
)

if label_col is None:
    raise Exception(
        f"Could not detect label column. Found columns: {email_df.columns.tolist()}"
    )


email_df = email_df[[email_col, label_col]].copy()
email_df.columns = ['text', 'label']

email_df['text'] = email_df['text'].fillna('').astype(str)

email_df['label'] = _clean_binary_labels(
    email_df['label'],
    {
        'spam': 1,
        'phishing': 1,
        'malicious': 1,
        '1': 1,
        'ham': 0,
        'legitimate': 0,
        'safe': 0,
        '0': 0,
        'bad': 1,
        'good': 0,
    },
)


# =====================================================
# URL DATASET COLUMN DETECTION
# =====================================================

url_col = next(
    (
        c
        for c in url_df.columns
        if 'url' in c.lower() or c.lower() == 'domain'
    ),
    None,
)

if url_col is None:
    raise Exception(
        f"Could not detect URL column. Found columns: {url_df.columns.tolist()}"
    )

label_col_url = next(
    (
        c
        for c in url_df.columns
        if 'label' in c.lower() or 'class' in c.lower()
    ),
    None,
)

if label_col_url is None:
    raise Exception(
        f"Could not detect URL label column. Found columns: {url_df.columns.tolist()}"
    )


url_df = url_df[[url_col, label_col_url]].copy()
url_df.columns = ['url', 'label']

url_df['url'] = url_df['url'].fillna('').astype(str)

url_df['label'] = _clean_binary_labels(
    url_df['label'],
    {
        'bad': 1,
        'phishing': 1,
        'malicious': 1,
        '1': 1,
        'good': 0,
        'benign': 0,
        'safe': 0,
        '0': 0,
    },
)


# =====================================================
# LIMIT DATA SIZE
# =====================================================

if len(email_df) > 5000:
    email_df = email_df.sample(5000, random_state=42)

if len(url_df) > 5000:
    url_df = url_df.sample(5000, random_state=42)


# =====================================================
# FEATURE EXTRACTION
# =====================================================

print("Extracting features...")

X = []
y = []

EXPECTED_SIZE = 39

EMAIL_ADDR_FEATS = safe_feature_vector(
    extract_email_address_features("user@example.com"),
    5,
)


# ---------------- EMAIL DATA ----------------

email_success = 0
email_failed = 0

for idx, row in email_df.iterrows():
    try:
        text = str(row['text']).strip()
        label = int(row['label'])

        if len(text) < 5:
            continue

        email_feats = safe_feature_vector(
            extract_email_text_features(text),
            9,
        )

        urls = extract_urls(text)

        url_feature_list = []

        for u in urls:
            try:
                feats = safe_feature_vector(
                    extract_url_features(u),
                    25,
                )
                url_feature_list.append(feats)
            except Exception:
                continue

        if url_feature_list:
            avg_url = np.mean(url_feature_list, axis=0)
        else:
            avg_url = np.zeros(25)

        features = np.concatenate(
            [
                email_feats,
                avg_url,
                EMAIL_ADDR_FEATS,
            ]
        )

        features = safe_feature_vector(features, EXPECTED_SIZE)

        X.append(features)
        y.append(label)

        email_success += 1

    except Exception as e:
        email_failed += 1
        continue


# ---------------- URL DATA ----------------

url_success = 0
url_failed = 0

for idx, row in url_df.iterrows():
    try:
        url = str(row['url']).strip()
        label = int(row['label'])

        if len(url) < 5:
            continue

        email_feats = safe_feature_vector(
            extract_email_text_features(url),
            9,
        )

        url_feats = safe_feature_vector(
            extract_url_features(url),
            25,
        )

        features = np.concatenate(
            [
                email_feats,
                url_feats,
                EMAIL_ADDR_FEATS,
            ]
        )

        features = safe_feature_vector(features, EXPECTED_SIZE)

        X.append(features)
        y.append(label)

        url_success += 1

    except Exception:
        url_failed += 1
        continue


# =====================================================
# DEBUG INFO
# =====================================================

print("\n========== DATA SUMMARY ==========")
print(f"Email success: {email_success}")
print(f"Email failed: {email_failed}")
print(f"URL success: {url_success}")
print(f"URL failed: {url_failed}")
print(f"Total training examples: {len(X)}")


if len(X) == 0:
    raise RuntimeError(
        "No training examples generated. Check feature_extraction.py functions."
    )


# =====================================================
# CONVERT TO ARRAYS
# =====================================================

X = np.array(X, dtype=float)
y = np.array(y, dtype=int)

print(f"Feature matrix shape: {X.shape}")
print(f"Labels shape: {y.shape}")


# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# =====================================================
# SCALE FEATURES
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =====================================================
# MODEL
# =====================================================

print("Training XGBoost model...")

model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric='logloss',
    random_state=42,
)


# =====================================================
# TRAIN
# =====================================================

model.fit(X_train_scaled, y_train)


# =====================================================
# EVALUATE
# =====================================================

pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, pred)

print(f"\nValidation Accuracy: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, pred))


# =====================================================
# SAVE
# =====================================================

model.save_model(MODEL_PATHS['xgb'])
joblib.dump(scaler, MODEL_PATHS['scaler'])

print("\nModel saved successfully!")
print(f"XGBoost model: {MODEL_PATHS['xgb']}")
print(f"Scaler: {MODEL_PATHS['scaler']}")
