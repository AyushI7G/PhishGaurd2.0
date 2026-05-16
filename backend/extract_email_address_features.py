import re
from collections import Counter
import numpy as np

def extract_email_address_features(email_addr):
    """
    Extract 8 phishing-related features from sender email address.
    Aligned with train.py logic.
    """
    email = str(email_addr)
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

