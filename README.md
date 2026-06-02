# OCULA

**Multilingual Multimodal Hate Speech Detection & Explainability**

> See the hate. Understand why. In Hindi/English. Stop the hate.

## What is OCULA?
OCULA detects hate speech in social media content across three classes:
**Hate / Offensive / Normal** - with support for English, Hindi, and Hinglish.
It explains *why* something was flagged using SHAP token-level explanations,
and runs as a Chrome browser extension in real-time.

## Stack
- Model: MuRIL-base-cased (multilingual, Indian languages)
- Explainability: SHAP + Attention Rollout
- API: FastAPI
- Extension: Chrome Manifest V3
