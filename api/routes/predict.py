import sys
import time 
import re
import logging
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
from fastapi import APIRouter,HTTPException

sys.path.insert(0,str(Path(__file__).parent.parent.parent))

from src.ocula.data.bridge_map import ID2LABEL
from api.schemas import PredictRequest,PredictResponse, TokenScore
from src.ocula.data.preprocess import TextCleaner

logger = logging.getLogger(__name__)
router = APIRouter()
_cleaner = TextCleaner(lowercase=True)

CHECKPOINT_DIR = Path("checkpoints/muril_base/checkpoint-25870")
MODEL_NAME = "google/muril-base-cased"
MAX_LENGTH = 128

model_state = {
    "model":None,
    "tokenizer":None,
    "device": None,
    "model_name":MODEL_NAME
}

def load_model():

    if model_state["model"] is not None:
        return
    
    logger.info(f"Loading Muril from {CHECKPOINT_DIR} ...")
    t0 = time.perf_counter()

    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    from src.ocula.models.classifier import OculaClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OculaClassifier(model_name=MODEL_NAME)

    weights_path = CHECKPOINT_DIR / "model.safetensors"

    state_dict = load_file(str(weights_path))
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model_state["model"] = model
    model_state["device"] = device
    model_state["tokenizer"] = tokenizer

    elapsed = round((time.perf_counter()- t0)*1000)
    logger.info(f"Model loaded in {elapsed}ms on {device}")


def clean_text(text):
    return _cleaner.clean(text)

def extract_word_importance(input_ids,attention_mask,attentions,tokenizer,org_text):
    last_layer = attentions[-1]
    avg_heads = last_layer[0].mean(dim=0)
    cls_row = avg_heads[0]
    cls_row =cls_row.cpu().float().numpy()

    ids = input_ids[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(ids)

    SKIP = {'[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'}

    words     = []
    buf_word  = ''
    buf_score = 0.0

    for token, score in zip(tokens, cls_row):
        if token in SKIP:
            # Flush buffer when hitting special token
            if buf_word:
                words.append((buf_word, buf_score))
                buf_word  = ''
                buf_score = 0.0
            continue

        if token.startswith('##'):
            # Continuation subword — append to current word
            buf_word  += token[2:]
            buf_score  = max(buf_score, float(score))
        else:
            # New word starts — flush previous word first
            if buf_word:
                words.append((buf_word, buf_score))
            buf_word  = token
            buf_score = float(score)
    
    if buf_word:
        words.append((buf_word, buf_score))
    
    if not words:
        return []
    
    scores = np.array([s for _,s in words],dtype=np.float32)
    s_min,s_max = scores.min(),scores.max()

    if s_max > s_min:
        scores = (scores-s_min)/(s_max-s_min)
    else:
        scores = np.ones_like(scores)
    
    text_lower = org_text.lower()
    search_from = 0
    result = []

    for i,((word,_),norm_score) in enumerate(zip(words,scores)):
        word_lower = word.lower()
        pos = text_lower.find(word_lower,search_from)
        if pos != -1:
            search_from = pos + len(word)
        
        result.append(TokenScore(word=word,score=round(float(norm_score),4),index=max(pos,0)))
    
    result.sort(key=lambda x: -x.score)
    return result


@router.post("/predict",response_model=PredictResponse)
async def predict(req:PredictRequest):
    t0 = time.perf_counter()
    load_model()

    model = model_state["model"]
    tokenizer = model_state["tokenizer"]
    device = model_state["device"]
    
    text = clean_text(req.text)
    if not text:
        raise HTTPException(status_code=400, detail="Empty text after cleaning")

    enc = tokenizer(text,return_tensors = "pt",max_length=MAX_LENGTH,truncation=True,padding=True)
    enc = {k:v.to(device) for k,v in enc.items()}

    with torch.no_grad():

        encoder_output = model.encoder(input_ids = enc["input_ids"],attention_mask=enc["attention_mask"],output_attentions = True)

        if hasattr(encoder_output,"pooler_output") and encoder_output.pooler_output is not None:
            pooled = encoder_output.pooler_output
        else:
            pooled = encoder_output.last_hidden_state[:,0,:]
        
        pooled = model.dropout(pooled)
        logits = model.classifier(pooled)

        probs = F.softmax(logits,dim=1).squeeze().cpu().numpy()

        label_id = int(np.argmax(probs))
        label = ID2LABEL[label_id]
        confidence = float(probs[label_id])
        probs_dict = {ID2LABEL[i]:round(float(probs[i]),4) for i in range(3)}

        top_tokens = extract_word_importance(input_ids=enc["input_ids"],attention_mask=enc["attention_mask"],attentions=encoder_output.attentions,tokenizer=tokenizer,org_text=text)

        latency = round((time.perf_counter() - t0)*1000,2)

        return PredictResponse(
            label=label,
            label_id=label_id,
            confidence=round(confidence,4),
            probabilities=probs_dict,
            top_tokens=top_tokens[:10],
            latency_ms = latency
        )