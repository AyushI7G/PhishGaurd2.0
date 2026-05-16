from flask import Flask, request, jsonify, send_from_directory
import xgboost as xgb
import numpy as np
import re
import os
import joblib
import warnings

warnings.filterwarnings('ignore')

from gmail_api import (
    build_gmail_service,
    list_unread_messages,
    get_message_payload,
    extract_email_body
)

from feature_extraction import (
    extract_email_text_features,
    extract_urls,
    extract_url_features,
    extract_email_address_features,
    fuse_features,
)

from settings import MODEL_PATHS, FEATURE_DIMENSIONS

app = Flask(__name__, static_folder='../frontend')

# =====================================================
# CORS
# =====================================================

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response


# =====================================================
# LOAD MODEL
# =====================================================

print("Loading model...")

xgb_model = xgb.XGBClassifier()

xgb_model.load_model(MODEL_PATHS['xgb'])

scaler = joblib.load(MODEL_PATHS['scaler'])

model = xgb_model

print("Model loaded successfully!")


# =====================================================
# HELPERS
# =====================================================

def clean_email_text(text):
    text = re.sub(r'\\s+', ' ', str(text))
    return text.strip()


def safe_vector(arr, size):
    arr = np.array(arr, dtype=float).flatten()

    if len(arr) < size:
        arr = np.concatenate([arr, np.zeros(size - len(arr))])

    elif len(arr) > size:
        arr = arr[:size]

    return arr


def classify_phishing(features):
    features = np.array(features, dtype=float).reshape(1, -1)

    features_scaled = scaler.transform(features)

    prob = model.predict_proba(features_scaled)[0][1]

    return float(prob)


# =====================================================
# ROUTES
# =====================================================

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok'
    })


# =====================================================
# EMAIL PREDICTION
# =====================================================

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():

    try:

        data = request.json

        email_text = clean_email_text(
            data.get('email_text', '')
        )

        email_address = data.get(
            'email_address',
            ''
        )

        # ---------------------------------------------
        # BASIC VALIDATION
        # ---------------------------------------------

        if not email_text.strip():

            return jsonify({
                'error': 'Please enter email content.'
            }), 400

        # ---------------------------------------------
        # FEATURE EXTRACTION
        # ---------------------------------------------

        email_feats = safe_vector(
            extract_email_text_features(email_text),
            FEATURE_DIMENSIONS['email_text']
        )

        urls = extract_urls(email_text)

        url_feature_list = []

        for u in urls:

            try:

                url_feature_list.append(
                    safe_vector(
                        extract_url_features(u),
                        FEATURE_DIMENSIONS['url']
                    )
                )

            except:
                continue

        if url_feature_list:

            url_feats = np.mean(
                url_feature_list,
                axis=0
            )

        else:

            url_feats = np.zeros(
                FEATURE_DIMENSIONS['url']
            )

        email_addr_feats = safe_vector(
            extract_email_address_features(email_address),
            FEATURE_DIMENSIONS['email_address']
        )

        fused = np.concatenate([
            email_feats,
            url_feats,
            email_addr_feats
        ])

        # ---------------------------------------------
        # MODEL PREDICTION
        # ---------------------------------------------

        prob = classify_phishing(fused)

        raw_score = round(prob * 100, 1)

        # ---------------------------------------------
        # CLASSIFICATION LOGIC
        # ---------------------------------------------

        if raw_score >= 95:

            classification = 'Phishing'

        elif raw_score >= 85:

            classification = 'Suspicious'

        else:

            classification = 'Legitimate'

        # ---------------------------------------------
        # DISPLAY RISK SCORE
        # ---------------------------------------------

        display_score = raw_score

        # Short email calibration
        if len(email_text.strip()) < 5:

            display_score = min(display_score, 45)

        elif len(email_text.strip()) < 15:

            display_score = min(display_score, 55)

        # UI consistency calibration
        if classification == 'Legitimate':

            display_score = min(display_score, 35)
            risk_level = 'Low'

        elif classification == 'Suspicious':

            display_score = max(
                min(display_score, 75),
                45
            )

            risk_level = 'Medium'

        else:

            display_score = max(display_score, 85)
            risk_level = 'High'

        # ---------------------------------------------
        # REASONING
        # ---------------------------------------------

        # ---------------------------------------------
        # AI ANALYSIS + SHAP STYLE FEATURES
        # ---------------------------------------------

        detected_features = []

        analysis_points = []

        lower_email = email_text.lower()

        # Suspicious keyword detection
        suspicious_words = [
            'urgent',
            'verify',
            'password',
            'login',
            'bank',
            'click',
            'security',
            'account',
            'suspend',
            'payment'
        ]

        matched_words = [
            w for w in suspicious_words
            if w in lower_email
        ]

        if matched_words:

            detected_features.append(
                f"Suspicious keywords detected: {', '.join(matched_words[:5])}"
            )

            analysis_points.append(
                "Urgency or credential-related language found."
            )

        # URL detection
        if 'http://' in lower_email or 'https://' in lower_email:

            detected_features.append(
                "External URL detected inside email body."
            )

            analysis_points.append(
                "Presence of links increases phishing probability."
            )

        # Uppercase detection
        uppercase_ratio = sum(
            1 for c in email_text if c.isupper()
        ) / max(len(email_text), 1)

        if uppercase_ratio > 0.25:

            detected_features.append(
                "High uppercase character usage detected."
            )

        # Exclamation mark detection
        if email_text.count('!') >= 3:

            detected_features.append(
                "Excessive exclamation marks detected."
            )

        # AI analysis generation
        if classification == 'Phishing':

            reasoning = (
                "The email demonstrates multiple phishing-related "
                "behavioral indicators including suspicious language "
                "patterns and potentially unsafe structures."
            )

        elif classification == 'Suspicious':

            reasoning = (
                "The email contains mixed indicators that may require "
                "manual verification before trusting the message."
            )

        else:

            reasoning = (
                "The email structure and language appear consistent "
                "with legitimate communication behavior."
            )

        # Fallback SHAP display
        if not detected_features:

            detected_features.append(
                "No major suspicious indicators detected."
            )

        # ---------------------------------------------
        # RESPONSE
        # ---------------------------------------------

        return jsonify({

            'probability': prob,

            'classification': classification,

            'risk_score': display_score,

            'risk_level': risk_level,

            'confidence_label': f'{display_score:.1f}%',

            'top_email_features': detected_features,

            'top_url_features': [],

            'top_email_address_features': [],

            'reasoning': reasoning,

            'ensemble_breakdown': {},

            'model_version': 'v2-stable'
        })

    except Exception as e:

        print("PREDICT ERROR:", str(e))

        return jsonify({
            'error': str(e)
        }), 500

# =====================================================
# URL PREDICTION
# =====================================================

@app.route('/predict_url', methods=['POST'])
def predict_url():

    try:

        data = request.json

        url = str(data.get('url', '')).strip()

        trusted_domains = [
            'google.com',
            'youtube.com',
            'microsoft.com',
            'github.com',
            'openai.com',
            'amazon.com',
            'paypal.com',
            'apple.com',
            'facebook.com',
            'instagram.com',
            'linkedin.com',
            'wikipedia.org'
        ]

        for domain in trusted_domains:

            if domain in url.lower():

                return jsonify({

                    'probability': 0.01,

                    'classification': 'Safe Website',

                    'risk_score': 1,

                    'confidence_label': '99%',

                    'top_url_features': {},

                    'reasoning':
                        'Trusted domain detected.',

                    'ensemble_breakdown': {},

                    'model_version': 'v2-stable'
                })

        if len(url) < 3:
            return jsonify({
                'error': 'Invalid URL'
            }), 400

        url_feats = safe_vector(
            extract_url_features(url),
            FEATURE_DIMENSIONS['url']
        )

        email_feats = np.zeros(
            FEATURE_DIMENSIONS['email_text']
        )

        email_addr_feats = np.zeros(
            FEATURE_DIMENSIONS['email_address']
        )

        fused = np.concatenate([
            email_feats,
            url_feats,
            email_addr_feats
        ])

        prob = classify_phishing(fused)

        classification = (
            'Phishing Website'
            if prob > 0.85
            else 'Safe Website'
        )

        risk_score = round(prob * 100, 1)

        return jsonify({
            'probability': prob,
            'classification': classification,
            'risk_score': risk_score,
            'confidence_label': f'{risk_score:.1f}%',

            'top_url_features': {},

            'reasoning': (
                'Suspicious URL patterns detected.'
                if classification == 'Phishing Website'
                else 'Website appears safe.'
            ),

            'ensemble_breakdown': {},
            'model_version': 'v2-stable'
        })

    except Exception as e:

        print("URL PREDICT ERROR:", str(e))

        return jsonify({
            'error': str(e)
        }), 500


# =====================================================
# START SERVER
# =====================================================

@app.route('/scan_gmail', methods=['GET'])
def scan_gmail():

    try:

        service = build_gmail_service(
            credentials_path='credentials.json'
        )

        messages = list_unread_messages(
            service,
            max_results=10
        )

        results = []

        for msg in messages:

            payload = get_message_payload(
                service,
                msg['id']
            )

            email_body = extract_email_body(payload)

            if not email_body:
                continue

            email_feats = safe_vector(
                extract_email_text_features(email_body),
                FEATURE_DIMENSIONS['email_text']
            )

            urls = extract_urls(email_body)

            url_feature_list = []

            for u in urls:

                try:

                    url_feature_list.append(
                        safe_vector(
                            extract_url_features(u),
                            FEATURE_DIMENSIONS['url']
                        )
                    )

                except:
                    continue

            if url_feature_list:

                url_feats = np.mean(
                    url_feature_list,
                    axis=0
                )

            else:

                url_feats = np.zeros(
                    FEATURE_DIMENSIONS['url']
                )

            email_addr_feats = np.zeros(
                FEATURE_DIMENSIONS['email_address']
            )

            fused = np.concatenate([
                email_feats,
                url_feats,
                email_addr_feats
            ])

            prob = classify_phishing(fused)

            if prob > 0.95:
                classification = 'Phishing'
                color = '#dc3545'

            elif prob > 0.9:
                classification = 'Suspicious'
                color = '#ffc107'

            else:
                classification = 'Legitimate'
                color = '#28a745'

            results.append({
                'snippet': (
                    email_body
                    .replace('\n', '<br>')
                    .replace('\r', '')
                    .strip()[:250]
                ),
                'classification': classification,
                'probability': round(prob * 100, 2),
                'color': color
            })

        # ============================================
        # BUILD HTML UI
        # ============================================

        html = """

        <html>

        <head>

        <title>PhishGuard Gmail Scan</title>

        <style>

        body{
            font-family:Arial;
            background:#0f172a;
            color:white;
            padding:30px;
        }

        h1{
            text-align:center;
            margin-bottom:40px;
        }

        .card{
            background:#1e293b;
            padding:20px;
            border-radius:12px;
            margin-bottom:20px;
            box-shadow:0 0 10px rgba(0,0,0,0.4);
        }

        .badge{
            padding:6px 14px;
            border-radius:8px;
            color:white;
            font-weight:bold;
            display:inline-block;
            margin-bottom:10px;
        }

        .prob{
            margin-top:10px;
            font-size:18px;
        }

        .snippet{
            margin-top:15px;
            line-height:1.6;
            color:#d1d5db;
            white-space:pre-wrap;
        }

        </style>

        </head>

        <body>

        <h1>📧 Gmail Inbox Scan Results</h1>

        """

        phishing_count = 0
        suspicious_count = 0
        legit_count = 0

        for r in results:

            if r['classification'] == 'Phishing':
                phishing_count += 1

            elif r['classification'] == 'Suspicious':
                suspicious_count += 1

            else:
                legit_count += 1

            html += f"""

            <div class="card">

                <div
                    class="badge"
                    style="background:{r['color']};"
                >
                    {r['classification']}
                </div>

                <div class="prob">
                    Risk Score:
                    {r['probability']}%
                </div>

                <div class="snippet">
                    {r['snippet']}
                </div>

            </div>

            """

        html = f"""

        <div style="
            display:flex;
            gap:20px;
            margin-bottom:40px;
            justify-content:center;
        ">

            <div class="card">
                ✅ Legitimate: {legit_count}
            </div>

            <div class="card">
                ⚠ Suspicious: {suspicious_count}
            </div>

            <div class="card">
                🚨 Phishing: {phishing_count}
            </div>

        </div>

        """ + html

        html += """

        </body>
        </html>

        """

        return html

    except Exception as e:

        print("GMAIL SCAN ERROR:", str(e))

        return f"""

        <h1>Gmail Scan Error</h1>

        <p>{str(e)}</p>

        """

if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
