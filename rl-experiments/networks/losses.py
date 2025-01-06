import torch
import torch.nn as nn
import torch.nn.functional as F


def explicit_InfoNCE(anchor, positive, negatives, temperature, hyperbolic=False, manifold=None, dot=False):
    """
    InfoNCE Loss using explicitly specified negative examples, instead of batching
    """
    if hyperbolic:
        positive_scores = -manifold.dist(x=anchor, y=positive).unsqueeze(1) / temperature
        negative_scores = -manifold.dist(x=anchor.unsqueeze(1), y=negatives) / temperature
    else:
        if dot:
            anchor = F.normalize(anchor, dim=-1)
            positive = F.normalize(positive, dim=-1)
            negatives = F.normalize(negatives, dim=-1)

            positive_scores = torch.sum(anchor * positive, dim=1, keepdim=True) / temperature
            negative_scores = torch.matmul(anchor.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1) / temperature
        else:
            positive_scores = -torch.sum((anchor - positive) ** 2, dim=1, keepdim=True) / temperature
            negative_scores = -torch.sum((anchor.unsqueeze(1) - negatives) ** 2, dim=2) / temperature

    logits = torch.cat([positive_scores, negative_scores], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

    return nn.CrossEntropyLoss()(logits, labels)


def symmetrized_InfoNCE(
    feature1_embed, feature2_embed, temperature, device, hyperbolic=False, manifold=None
):
    """
    InfoNCE Loss using random pairs from other batched examples as negative pairs.
    """
    # Compute pairwise L2 distances
    if hyperbolic:
        dist_matrix = manifold.dist(
            x=feature1_embed.unsqueeze(1), y=feature2_embed.unsqueeze(0)
        )
    else:
        dist_matrix = torch.cdist(feature1_embed, feature2_embed, p=2.0)

    # Convert distances to similarities (negative distances)
    logits = -dist_matrix / temperature

    labels = torch.arange(feature1_embed.shape[0], device=device)
    loss_1 = F.cross_entropy(logits, labels)
    loss_2 = F.cross_entropy(logits.t(), labels)
    return (loss_1 + loss_2) / 2


