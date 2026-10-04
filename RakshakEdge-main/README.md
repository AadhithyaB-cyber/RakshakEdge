---
language:
- en
pipeline_tag: text-classification
tags:
- sms-spam
- phishing-detection
- scam-detection
- security
metrics:
- f1
- accuracy
- precision
- recall
widget:
- text: "Your account is blocked! Verify immediately with OTP. Send money to scam@ybl using https://scam.xyz/"
  example_title: "Bank KYC Scam"
- text: "Congratulations! You won Rs 50,000 lottery prize. Contact urgently to claim via link: http://bit.ly/claim"
  example_title: "Lottery Scam"
- text: "Hey, are we still meeting for lunch tomorrow at 12?"
  example_title: "Safe Message"
---

# SCAMBERT: DistilBERT for SMS Fraud & Scam Detection

SCAMBERT is a fine-tuned `distilbert-base-uncased` model specifically designed to detect social engineering, financial fraud, phishing, and scam payloads in SMS and short-form conversational text. It is built as Layer 3 of the AI Honeypot (CIPHER) Threat Intelligence Pipeline.

## Model Summary

- **Model Type:** Text Classification (Binary)
- **Base Model:** `distilbert-base-uncased`
- **Language:** English (en)
- **Task:** Spam/Scam Detection
- **License:** MIT (or your project's license)
- **Size:** ~255 MB

### Labels

- `0`: Safe / Legitimate
- `1`: Scam / Fraud / Phishing

## Performance & Metrics

The model was fine-tuned on a dataset of **8,438** samples (27.5% Scam / 72.5% Safe). Due to class imbalance, class weights were applied during training.

### Calibration & Validation Results

- **Best Accuracy:** 99.41%
- **Best F1-Score:** 98.92%
- **Calibrated Precision:** 95.08%
- **Calibrated Recall:** 100.0%
- **Optimal Threshold:** `0.0028` (For high-recall environments)

### Robustness Evaluation

The model was tested against common bad-actor obfuscation tactics:

| Tactic | Example Input | Prediction Probability | Passed |
| :--- | :--- | :--- | :--- |
| **URL Obfuscation** | `Win $1000 fast! Click hxxp://scammy...` | 99.9% Scam | ✅ |
| **Numeric Substitution** | `W1NNER! Y0u have b33n select3d...` | 99.3% Scam | ✅ |
| **Mixed Case** | `cOnGrAtUlAtIoNs, yOu WoN a FrEe...` | 89.8% Scam | ✅ |

*Note: The model occasionally struggles with extremely short, contextless messages (e.g., "Call me now") as intended, relying on earlier heuristic layers for context.*

## Usage

You can use this model directly with Hugging Face's `pipeline`:

```python
from transformers import pipeline

# Load the pipeline
classifier = pipeline("text-classification", model="Digvijay05/SCAMBERT")

# Inference
text = "Earn Rs 5000 daily income from home part time. Click this link: http://bit.ly/job"
result = classifier(text)

print(result)
# [{'label': 'LABEL_1', 'score': 0.99...}] 
```

Or run via the **Inference API**:

```python
import httpx

API_URL = "https://api-inference.huggingface.co/models/Digvijay05/SCAMBERT"
headers = {"Authorization": "Bearer YOUR_HF_TOKEN"}

response = httpx.post(API_URL, headers=headers, json={"inputs": "Your account is locked. Verify at bit.ly/secure"})
print(response.json())
```

## Deployment Considerations

- **CPU Latency Estimate:** ~10-30ms / sequence
- **GPU Latency Estimate:** ~2-5ms / sequence
- **Recommendation:** Can be efficiently hosted on serverless CPU environments (like Render Free Tier) using Hugging Face's Inference API, or deployed natively if 512MB+ RAM is available. ONNX quantization is recommended for edge deployments.

## Intended Use

This model is designed as a *semantic booster* tie-breaker layer within a multi-layered classification engine. It excels at detecting complex sentence structures, urgency, and manipulative context that standard Regex/Heuristic rules might miss.
