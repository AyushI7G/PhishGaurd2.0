from flask import Flask, request, jsonify, send_from_directory
import xgboost as xgb
import shap
import numpy as np
import pandas as pd
import re
from urllib.parse import urlparse
import string
import joblib
import os

app = Flask(__name__, static_folder='../frontend')

# Enable CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Load model
script_dir = os.path.dirname(os.path.abspath(__file__))
xgb_model = xgb.Booster()
xgb_model.load_model(os.path.join(script_dir, 'xgb_model.json'))
scaler = joblib.load(os.path.join(script_dir, 'scaler.pkl'))

# -----------------------------
# TEXT PROCESSING
# -----------------------------

def clean_email_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_urls(text):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)

def extract_email_text_features(text):
    features = {}

    text_lower = text.lower()

    features['email_length'] = len(text)
    features['num_urls'] = len(extract_urls(text))
    features['num_uppercase'] = sum(1 for c in text if c.isupper())
    features['num_exclamations'] = text.count('!')
    features['num_digits'] = sum(c.isdigit() for c in text)

    suspicious_words = ['urgent', 'verify', 'password', 'click', 'account', 'bank', 'login']
    features['suspicious_word_count'] = sum(word in text_lower for word in suspicious_words)

    return list(features.values())

# -----------------------------
# URL FEATURES
# -----------------------------

def extract_url_features(url):
    features = {}
    parsed = urlparse(url)

    features['url_length'] = len(url)

    domain_parts = parsed.netloc.split('.')
    features['num_subdomains'] = len(domain_parts) - 1 if len(domain_parts) > 1 else 0

    ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    features['has_ip'] = 1 if re.search(ip_pattern, parsed.netloc) else 0

    features['is_https'] = 1 if parsed.scheme == 'https' else 0

    features['special_chars'] = sum(1 for char in url if char in string.punctuation)

    from collections import Counter
    char_counts = Counter(url)
    total_chars = len(url)
    entropy = -sum((count / total_chars) * np.log2(count / total_chars) for count in char_counts.values())
    features['entropy'] = entropy

    features['path_length'] = len(parsed.path)
    features['query_length'] = len(parsed.query)
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['num_letters'] = sum(c.isalpha() for c in url)
    features['has_at'] = 1 if '@' in url else 0
    features['has_hyphen'] = 1 if '-' in parsed.netloc else 0
    features['tld_length'] = len(parsed.netloc.split('.')[-1]) if '.' in parsed.netloc else 0

    for i in range(25 - len(features)):
        features[f'feature_{i}'] = 0

    return list(features.values())[:25]

# -----------------------------
# EMAIL ADDRESS FEATURES
# -----------------------------

from extract_email_address_features import extract_email_address_features

# -----------------------------
# FEATURE FUSION
# -----------------------------

def fuse_features(email_text_feats, url_features_list, email_addr_feats):
    if url_features_list:
        avg_url_features = np.mean(url_features_list, axis=0)
    else:
        avg_url_features = np.zeros(25)

    return np.concatenate([email_text_feats, avg_url_features, email_addr_feats])

# -----------------------------
# MODEL PREDICTION
# -----------------------------

def classify_phishing(features):
    features_scaled = scaler.transform([features])
    dmatrix = xgb.DMatrix(features_scaled)
    return xgb_model.predict(dmatrix)[0]

FEATURE_NAMES = [
    'email_length', 'num_urls', 'num_uppercase', 'num_exclamations',
    'num_digits', 'suspicious_word_count'
] + [
    'url_length', 'num_subdomains', 'has_ip', 'is_https', 'special_chars', 'entropy',
    'path_length', 'query_length', 'num_digits_url', 'num_letters_url',
    'has_at', 'has_hyphen', 'tld_length'
] + [f'feature_{i}' for i in range(25 - 13)] + [
    'email_addr_len', 'domain_len', 'suspicious_tld', 'free_domain',
    'num_dots_domain', 'has_digits_addr', 'domain_entropy', 'has_plus'
]

def get_shap_explanation(features):
    features_scaled = scaler.transform([features])
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(features_scaled)

    abs_vals = np.abs(shap_values[0])
    top_indices = np.argsort(abs_vals)[-10:][::-1]

    return {FEATURE_NAMES[i]: float(shap_values[0][i]) for i in top_indices}

# -----------------------------
# ROUTES
# -----------------------------

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    email_text = clean_email_text(data.get('email_text', ''))
    text_lower = email_text.lower()

    suspicious_words = ['click', 'verify', 'password', 'login', 'pay', 'urgent']

    # Handle short emails safely
    if len(email_text.strip()) < 25:
        if not any(word in text_lower for word in suspicious_words):
            return jsonify({
                'probability': 0.2,
                'classification': 'Legitimate',
                'top_features': {}
            })
    email_address = data.get('email_address', '')

    email_text_feats = extract_email_text_features(email_text)

    urls = extract_urls(email_text)
    url_features = [extract_url_features(url) for url in urls]

    email_addr_feats = extract_email_address_features(email_address)

    fused = fuse_features(email_text_feats, url_features, email_addr_feats)

    prob = classify_phishing(fused)
    classification = 'Phishing' if prob > 0.7 else 'Legitimate'

    shap_data = get_shap_explanation(fused)

    return jsonify({
        'probability': float(prob),
        'classification': classification,
        'top_features': shap_data
    })

@app.route('/predict_url', methods=['POST'])
def predict_url():
    data = request.json
    url = data.get('url', '')

    # Extract URL features
    url_feats = extract_url_features(url)

    # Empty email + sender features (to match model input size)
    email_text_feats = np.zeros(6)
    email_addr_feats = np.zeros(8)

    fused = np.concatenate([email_text_feats, url_feats, email_addr_feats])

    prob = classify_phishing(fused)
    classification = 'Phishing Website' if prob > 0.5 else 'Safe Website'

    return jsonify({
        'probability': float(prob),
        'classification': classification
    })

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)