import sys
import time
import logging
import html as html_lib
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from fastapi import APIRouter,HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.schemas import ExplainRequest, ExplainResponse, TokenScore
from api.routes.predict import clean_text,MAX_LENGTH,get_model_for_text

from src.ocula.data.bridge_map import ID2LABEL

logger = logging.getLogger(__name__)
router = APIRouter()


def run_shap(text,top_k,model,tokenizer,device):

    def predict(texts):
        texts = [str(t) for t in texts]
        all_probs = []
        batch_size = 8
        for i in range(0,len(texts),8):
            batch = texts[i: i+batch_size]
            enc = tokenizer(batch,return_tensors="pt",max_length=MAX_LENGTH,truncation=True,padding=True)
            enc = {k:v.to(device) for k,v in enc.items()}

            with torch.no_grad():
                out = model.encoder(input_ids = enc["input_ids"],attention_mask = enc["attention_mask"])
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    pooled = out.pooler_output
                else:
                    pooled = out.last_hidden_state[:, 0, :]

                pooled = model.dropout(pooled)
                logits = model.classifier(pooled)
                probs  = F.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        return np.vstack(all_probs)
    
    import shap

    masker = shap.maskers.Text(r"\s+")
    explainer = shap.Explainer(predict,masker,output_names=list(ID2LABEL.values()))
    shap_values = explainer([text], fixed_context=1, nsamples=50)

    probs_base = predict([text])[0]
    label_id = int(np.argmax(probs_base))
    label = ID2LABEL[label_id]
    confidence = float(probs_base[label_id])
    prob_dict = {ID2LABEL[i]: round(float(probs_base[i]), 4) for i in range(3)}

    token_vals = shap_values.values[0,:,label_id]
    token_words = shap_values.data[0]

    abs_vals = np.abs(token_vals)
    v_min, v_max = abs_vals.min(), abs_vals.max()
    if v_max > v_min:
        norm_scores = (abs_vals-v_min)/(v_max-v_min)
    else:
        norm_scores = np.ones_like(abs_vals)
    
    text_lower  = text.lower()
    search_from = 0
    token_scores = []

    for word, raw_val, norm_score in zip(token_words, token_vals, norm_scores):
        word_str= str(word).strip()
        word_lower=word_str.lower()
        if not word_str:
            continue

        pos = text_lower.find(word_lower, search_from)
        if pos!= -1:
            search_from = pos + len(word_str)

        token_scores.append(TokenScore(
            word  = word_str,
            score = round(float(norm_score), 4),
            index = max(pos, 0),
        ))

    token_scores.sort(key=lambda x: -x.score)

    html_highlight = build_html_highlight(text, token_scores, label_id)
    return token_scores[:top_k],html_highlight, label, label_id, confidence, prob_dict

def build_html_highlight(text,token_scores,label_id):
    label_class = {0: "hate", 1: "offensive", 2: "normal"}[label_id]
    intensity = {
        0.7:  "ocula-high",
        0.4:  "ocula-med",
        0.2:  "ocula-low",
    }
    word_scores: dict[str, float] = {}
    for ts in token_scores:
        w = ts.word.lower()
        if w not in word_scores or ts.score > word_scores[w]:
            word_scores[w] = ts.score
    
    result_parts = []
    words = text.split(" ")

    for word in words:
        clean_w = word.lower().strip(".,!?;:\"'()[]{}—-")
        score   = word_scores.get(clean_w, 0.0)
        css_intensity = None
        for threshold in sorted(intensity.keys(), reverse=True):
            if score >= threshold:
                css_intensity = intensity[threshold]
                break

        if css_intensity:
            safe_word = html_lib.escape(word)
            result_parts.append(
                f'<span class="{css_intensity} {label_class}" '
                f'data-score="{score:.3f}" '
                f'title="SHAP score: {score:.3f}">'
                f'{safe_word}</span>'
            )
        else:
            result_parts.append(html_lib.escape(word))

    return " ".join(result_parts)


@router.post("/explain", response_model=ExplainResponse)
async def explain(req: ExplainRequest):

    text = clean_text(req.text)
    if not text:
        raise HTTPException(status_code=400, detail="Empty text after cleaning")

    try:
        model, tokenizer, device, model_key = get_model_for_text(text)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        token_scores,html_highlight,label,label_id,confidence,prob_dict = run_shap(
            text, top_k=req.top_k, model=model, tokenizer=tokenizer, device=device
        )
    except Exception as e:
        logger.error(f"SHAP failed: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

    return ExplainResponse(
        label          = label,
        label_id       = label_id,
        confidence     = round(confidence, 4),
        probabilities  = prob_dict,
        tokens         = token_scores,
        highlight  = html_highlight,
    )