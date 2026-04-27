import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import re
from urllib.parse import urlparse
import string
from collections import Counter

# -----------------------------
# EMAIL TEXT FEATURES
# -----------------------------

def extract_urls(text):
    url_pattern = r'http[s]?://(?:[a-zA-Z0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, str(text))

def extract_email_text_features(text):
    text = str(text)
    text_lower = text.lower()

    return [
        len(text),
        len(extract_urls(text)),
        sum(1 for c in text if c.isupper()),
        text.count('!'),
        sum(c.isdigit() for c in text),
        sum(word in text_lower for word in ['urgent','verify','password','click','account','bank','login'])
    ]

# -----------------------------
# URL FEATURES
# -----------------------------

def extract_url_features(url):
    url = str(url)
    parsed = urlparse(url)

    features = []

    features.append(len(url))

    domain_parts = parsed.netloc.split('.')
    features.append(len(domain_parts) - 1 if len(domain_parts) > 1 else 0)

    ip_pattern = r'\b\d{1,3}(\.\d{1,3}){3}\b'
    features.append(1 if re.search(ip_pattern, parsed.netloc) else 0)

    features.append(1 if parsed.scheme == 'https' else 0)

    features.append(sum(1 for c in url if c in string.punctuation))

    char_counts = Counter(url)
    total_chars = len(url)
    entropy = -sum((count / total_chars) * np.log2(count / total_chars) for count in char_counts.values()) if total_chars else 0
    features.append(entropy)

    features.append(len(parsed.path))
    features.append(len(parsed.query))
    features.append(sum(c.isdigit() for c in url))
    features.append(sum(c.isalpha() for c in url))
    features.append(1 if '@' in url else 0)
    features.append(1 if '-' in parsed.netloc else 0)
    features.append(len(parsed.netloc.split('.')[-1]) if '.' in parsed.netloc else 0)

    # pad to 25
    features += [0] * (25 - len(features))

    return features[:25]

# -----------------------------
# EMAIL ADDRESS FEATURES
# -----------------------------

def extract_email_address_features(email):
    email = str(email)
    domain = email.split('@')[-1] if '@' in email else ''

    return [
        len(email),
        len(domain),
        int(domain.endswith(('.ru', '.tk', '.ml', '.ga'))),
        int(domain in ['gmail.com','yahoo.com','outlook.com']),
        domain.count('.'),
        int(any(c.isdigit() for c in email)),
        len(set(domain)) / (len(domain)+1),
        int('+' in email)
    ]

# -----------------------------
# LOAD DATASETS
# -----------------------------

# Email dataset
email_df = pd.read_csv('../dataset/CEAS_08.csv')

# Try to detect columns automatically
email_col = None
for col in email_df.columns:
    if 'body' in col.lower():
        email_col = col
        break

if email_col is None:
    raise Exception("Email body column not found in CEAS_08.csv")

label_col = None
for col in email_df.columns:
    if 'label' in col.lower():
        label_col = col
        break

if label_col is None:
    raise Exception("Label column not found in CEAS_08.csv")

email_df = email_df[[email_col, label_col]].dropna()
email_df.columns = ['text', 'label']

# Convert labels (spam = 1, ham = 0)
email_df['label'] = email_df['label'].astype(str).str.lower().map({
    'spam':1, 'phishing':1, '1':1,
    'ham':0, 'legitimate':0, '0':0
}).fillna(0)

email_df = email_df.sample(n=min(5000, len(email_df)), random_state=42)

# URL dataset
url_df = pd.read_csv('../dataset/urldata.csv')
url_df = url_df.sample(n=5000, random_state=42)

url_df['label'] = url_df['label'].map({'bad':1, 'good':0})

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------

X = []
y = []

# EMAIL DATA
for _, row in email_df.iterrows():
    text = row['text']
    label = row['label']

    email_feats = extract_email_text_features(text)
    urls = extract_urls(text)
    url_feats = [extract_url_features(u) for u in urls]

    if url_feats:
        avg_url = np.mean(url_feats, axis=0)
    else:
        avg_url = np.zeros(25)

    email_addr_feats = extract_email_address_features("user@example.com")

    features = np.concatenate([email_feats, avg_url, email_addr_feats])

    X.append(features)
    y.append(label)

# URL DATA
for _, row in url_df.iterrows():
    url = row['url']
    label = row['label']

    email_feats = extract_email_text_features(url)  # simulate
    url_feats = extract_url_features(url)
    email_addr_feats = extract_email_address_features("user@example.com")

    features = np.concatenate([email_feats, url_feats, email_addr_feats])

    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

# -----------------------------
# TRAIN
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100)
model.fit(X_train_scaled, y_train)

# -----------------------------
# SAVE
# -----------------------------

model.save_model('xgb_model.json')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model trained with EMAIL + URL features")