import torch.nn.functional as F
import torch
def edge_attention_loss(preds, targets, edges, beta=2, alpha=1.5):
    base_loss = F.cross_entropy(preds, targets, reduction='mean')

    attention_weights = torch.ones_like(edges) + alpha * edges


    edge_loss = F.cross_entropy(preds, targets, reduction='none')
    edge_loss = (edge_loss * attention_weights).mean()


    total_loss = base_loss + beta * edge_loss


    return total_loss