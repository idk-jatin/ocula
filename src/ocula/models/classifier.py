import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from src.ocula.data.bridge_map import NUM_LABELS,LABEL2ID,ID2LABEL

class OculaClassifier(nn.Module):
    def __init__(self,model_name="google/muril-base-cased",num_labels=NUM_LABELS,dropout=0.01):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

        config = AutoConfig.from_pretrained(model_name,num_labels=num_labels,id2label=ID2LABEL,label2id=LABEL2ID)
        self.encoder = AutoModel.from_pretrained(model_name,config=config)
        hidden_size = config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size,num_labels)
    
    def gradient_checkpointing_enable(self,gradient_checkpointing_kwargs=None):
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        self.encoder.gradient_checkpointing_disable()

    def forward(self,input_ids,attention_mask,token_type_ids=None,labels=None):

        encoder_kwargs = {"input_ids":input_ids,"attention_mask":attention_mask}

        if (token_type_ids is not None and "distilbert" not in self.model_name.lower() and "xlm-roberta" not in self.model_name.lower()):
            encoder_kwargs["token_type_ids"] = token_type_ids
        
        outputs = self.encoder(**encoder_kwargs)
        if (hasattr(outputs,"pooler_output") and outputs.pooler_output is not None):
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:,0,:]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        result = {"logits":logits}

        if labels is not None:
            from src.ocula.models.focal_loss import FocalLoss
            loss_fn = FocalLoss(gamma=2.0,num_classes=self.num_labels)
            result['loss'] = loss_fn(logits,labels)
        return result
    
