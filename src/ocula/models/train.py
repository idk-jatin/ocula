import os
import sys
import json
import logging
import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.ocula.models.classifier import OculaClassifier
from src.ocula.models.dataset import OculaDataset
from src.ocula.models.focal_loss import compute_alpha_weights
from src.ocula.data.bridge_map import ID2LABEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_compute_metrics(id2label):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        macro_f1 = f1_score(labels, predictions, average="macro")
        weighted_f1 = f1_score(labels, predictions, average="weighted")
        per_class = f1_score(
            labels, predictions, average=None, labels=list(id2label.keys())
        )
        return {
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "f1_hate": round(per_class[0], 4),
            "f1_offensive": round(per_class[1], 4),
            "f1_normal": round(per_class[2], 4),
        }

    return compute_metrics


def train(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    num_labels = cfg["model"]["num_labels"]
    dropout = cfg["model"]["dropout"]
    output_dir = cfg["training"]["output_dir"]
    max_length = cfg["data"]["max_length"]
    train_path = cfg["data"]["train_path"]
    val_path = cfg["data"]["val_path"]
    test_path = cfg["data"]["test_path"]
    gamma = cfg["focal_loss"]["gamma"]

    logger.info(f"Config: {config_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Model: {model_name}")

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Loading Dataset...")
    train_dataset = OculaDataset(train_path,tokenizer,max_length)
    test_dataset = OculaDataset(test_path,tokenizer,max_length)
    val_dataset = OculaDataset(val_path,tokenizer,max_length)

    logger.info(f"Train: {len(train_dataset):,}")
    logger.info(f"Val:   {len(val_dataset):,}")
    logger.info(f"Test:  {len(test_dataset):,}")

    label_counts = train_dataset.get_label_counts()
    logger.info(f"Training label counts: {label_counts}")

    if cfg["focal_loss"]["alpha"] is not None:
        alpha = cfg["focal_loss"]["alpha"]
        logger.info(f"Using config alpha: {alpha}")
    else:
        alpha = compute_alpha_weights(label_counts)
        logger.info(f"Computed alpha weights: {alpha}")
    
    logger.info(f"Loading model: {model_name}")
    model = OculaClassifier(model_name=model_name,num_labels=num_labels,dropout=dropout)
    model._focal_alpha = alpha
    model._focal_gamma = gamma

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg["training"]["num_train_epochs"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        warmup_ratio = cfg["training"]["warmup_ratio"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        max_grad_norm=cfg["training"]["max_grad_norm"],
        fp16= torch.cuda.is_available() and cfg["training"]["fp16"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        eval_strategy = "epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=cfg["training"]["logging_steps"],
        report_to=cfg["training"]["report_to"],
        dataloader_num_workers=2,
        seed = 42,
        save_total_limit=2,
    )

    class OculaTrainer(Trainer):
        def compute_loss(self,model,inputs,return_outputs=False,**kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs["logits"]
            from src.ocula.models.focal_loss import FocalLoss
            loss_fn = FocalLoss(gamma=model._focal_gamma,alpha=model._focal_alpha,num_classes=model.num_labels)
            loss = loss_fn(logits,labels)
            return (loss,outputs) if return_outputs else loss
        
        def _save(self,output_dir=None,state_dict=None):
            if state_dict is None:
                state_dict = self.model.state_dict()
            
            state_dict = {k:v.contiguous() for k,v in state_dict.items()}
            super()._save(output_dir,state_dict)


    trainer = OculaTrainer(model=model,args=training_args,train_dataset=train_dataset,eval_dataset=val_dataset,compute_metrics=build_compute_metrics(ID2LABEL),callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])

    logger.info("Starting Training...")
    trainer.train(resume_from_checkpoint=True)

    logger.info("Evaluating on test set...")
    predictions = trainer.predict(test_dataset)

    pred_labels = np.argmax(predictions.predictions,axis=-1)
    true_labels = predictions.label_ids

    report = classification_report(true_labels,pred_labels,target_names=[ID2LABEL[i] for i in range(3)],digits=4)
    logger.info(f"\n{report}")

    cm = confusion_matrix(true_labels,pred_labels)
    logger.info(f"Confusion matrix:\n{cm}")

    results = {
        "model":model_name,
        "macro_f1":round(f1_score(true_labels, pred_labels, average="macro"), 4),
        "weighted_f1":round(f1_score(true_labels, pred_labels, average="weighted"), 4),
        "per_class_f1": { ID2LABEL[i]: round(v, 4) for i, v in enumerate(f1_score(true_labels, pred_labels, average=None)) },
        "confusion_matrix": cm.tolist(),
        "report": report,
        "alpha_weights": alpha,
    }

    results_path = Path(output_dir) / "results.json"
    results_path.parent.mkdir(parents=True,exist_ok=True)
    with open(results_path,"w") as f:
        json.dump(results,f,indent=2)
    
    logger.info(f"Results saved to {results_path}")
    logger.info(f"Macro-F1: {results['macro_f1']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",required=True,help="Path to config YAML file")
    args = parser.parse_args()
    train(args.config)