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

MAX_LENGTH = 128

MODELS_CONFIG = {
    "muril": {
        "checkpoint": Path("checkpoints/muril_base/checkpoint-25870"),
        "model_name": "google/muril-base-cased",
    },
    "xlm_roberta": {
        "checkpoint": Path("checkpoints/xlm_roberta/checkpoint-15522"),
        "model_name": "xlm-roberta-base",
    }
}

model_state = {
    "muril": {
        "model": None,
        "tokenizer": None,
        "device": None
    },
    "xlm_roberta": {
        "model": None,
        "tokenizer": None,
        "device": None
    }
}

def detect_language_route(text):
    text_clean = text.strip()
    if not text_clean:
        return "xlm_roberta"
        
    # 1. Devanagari character detection -> MuRIL
    if re.search(r"[\u0900-\u097F]", text_clean):
        return "muril"
        
    # 2. Check for common Hinglish particles/pronouns (routes immediately to MuRIL)
    words = set(re.findall(r"\b[a-z]+\b", text_clean.lower()))
    HINGLISH_PARTICLES = {
        "hai", "hain", "hoon", "hu", "ho", "ka", "ki", "ke", "ko", "se", "pe",
        "aur", "bhi", "toh", "ya", "kya", "kab", "kyun", "kyu", "kaha",
        "kidhar", "kaise", "ye", "yeh", "wo", "woh", "jo", "tum", "aap", "mujhe",
        "mera", "meri", "apna", "apni", "tere", "tera", "teri", "bhai", "yaar",
        "nahi", "nahin", "nhi", "tha", "thi", "raha", "rha", "rahi", "rhi", "rahe",
        "rhe", "karna", "karo", "kar", "kiya", "diya", "liye", "chalo", "achha",
        "acha", "ganda", "badhiya", "chor"
    }
    if words.intersection(HINGLISH_PARTICLES):
        return "muril"
        
    # 3. Probability check using langdetect (if English is present with >10% probability)
    try:
        from langdetect import detect_langs, DetectorFactory
        DetectorFactory.seed = 0
        langs = detect_langs(text_clean)
        for l in langs:
            if l.lang == "en" and l.prob > 0.1:
                return "xlm_roberta"
    except Exception:
        pass
        
    return "muril"

def load_model(model_key="muril"):
    state = model_state[model_key]
    if state["model"] is not None:
        return state["model"], state["tokenizer"], state["device"]
        
    config = MODELS_CONFIG[model_key]
    checkpoint_dir = config["checkpoint"]
    model_name = config["model_name"]
    
    logger.info(f"Loading {model_key} model ({model_name}) from {checkpoint_dir} ...")
    t0 = time.perf_counter()

    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    from src.ocula.models.classifier import OculaClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OculaClassifier(model_name=model_name)

    weights_path = checkpoint_dir / "model.safetensors"

    state_dict = load_file(str(weights_path))
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    state["model"] = model
    state["device"] = device
    state["tokenizer"] = tokenizer

    elapsed = round((time.perf_counter()- t0)*1000)
    logger.info(f"{model_key} model loaded in {elapsed}ms on {device}")
    
    return model, tokenizer, device

def get_model_for_text(text):
    model_key = detect_language_route(text)
    log_msg = f"[ROUTING] Text: '{text[:60]}...' -> Selected Model: {model_key.upper()}"
    print(log_msg, flush=True)
    logger.info(log_msg)
    model, tokenizer, device = load_model(model_key)
    return model, tokenizer, device, model_key


def clean_text(text):
    return _cleaner.clean(text)

def extract_word_importance(input_ids,attention_mask,attentions,tokenizer,org_text):
    last_layer = attentions[-1]
    avg_heads = last_layer[0].mean(dim=0)
    cls_row = avg_heads[0]
    cls_row =cls_row.cpu().float().numpy()

    ids = input_ids[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(ids)

    SKIP = {'[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]', '<s>', '</s>', '<pad>', '<unk>', '<mask>'}

    # Detect if SentencePiece (e.g., xlm-roberta) is used
    is_sentencepiece = any(t.startswith('\u2581') for t in tokens if t not in SKIP)

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

        if is_sentencepiece:
            if token.startswith('\u2581'):
                # New word starts
                if buf_word:
                    words.append((buf_word, buf_score))
                buf_word  = token[1:]  # strip the SentencePiece prefix
                buf_score = float(score)
            else:
                # Continuation subword
                buf_word  += token
                buf_score = max(buf_score, float(score))
        else:
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
    
    text = clean_text(req.text)
    if not text:
        raise HTTPException(status_code=400, detail="Empty text after cleaning")

    model, tokenizer, device, model_key = get_model_for_text(text)

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