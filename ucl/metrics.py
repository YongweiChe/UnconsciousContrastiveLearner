import torch


def recall_at_k(scores: torch.Tensor, k: int = 1) -> float:
    """Fraction of rows whose positive (the diagonal entry) ranks in the top k.

    Ties are broken uniformly at random (in expectation): a positive tied with t - 1 other
    candidates, below g strictly better ones, is in the top k with probability
    clamp((k - g) / t, 0, 1). A constant score matrix therefore scores chance, k / n.
    """
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError(f"expected a square score matrix, got {tuple(scores.shape)}")
    if not torch.isfinite(scores).all():
        raise ValueError("score matrix contains non-finite values")
    pos = scores.diagonal()[:, None]
    better = (scores > pos).sum(1).float()
    tied = (scores == pos).sum(1).float()  # includes the positive itself
    return ((k - better) / tied).clamp(0, 1).mean().item()
