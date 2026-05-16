import torch
import torch.nn as nn 
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self,gamma=2,alpha=None,reduction="mean",num_classes=3):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

        if alpha is not None:
            self.register_buffer("alpha",torch.tensor(alpha,dtype=torch.float32))
        else:
             self.register_buffer("alpha",torch.ones(num_classes))


    def forward(self,logits,targets):
        ce_loss = F.cross_entropy(logits,targets,reduction="none")

        probs = F.softmax(logits,dim=-1)
        p_t = probs.gather(1,targets.unsqueeze(1)).squeeze(1)
        alpha = self.alpha.to(targets.device)
        alpha_t = alpha.gather(0,targets)
        focal_weight = (1.0 - p_t)**self.gamma
        loss = alpha_t*focal_weight*ce_loss

        if self.reduction =="mean":
            return loss.mean()
        elif self.reduction=="sum":
            return loss.sum()
        return loss


def compute_alpha_weights(label_cnt,num_classes=3):
    total = sum(label_cnt.values())

    weights = []
    for i in range(num_classes):
        count = label_cnt.get(i,1)
        weights.append(total/(num_classes*count))

    mean_w = sum(weights)/len(weights)
    weights = [round(w/mean_w,4) for w in weights]
    return weights
