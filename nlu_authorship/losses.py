"""The project's historical binary focal adaptation, evaluated against CE.

This is not the paper's multilabel sigmoid ASL or its ASLSingleLabel recipe.
The true class receives gamma=2 for different-author and gamma=1 for same-author.
"""
from __future__ import annotations


def binary_asymmetric_focal_loss(logits, labels, gamma_neg=2.0, gamma_pos=1.0, reduction="mean"):
    import torch
    import torch.nn.functional as F

    if logits.ndim != 2 or logits.shape[1] != 2 or labels.ndim != 1 or len(labels) != len(logits):
        raise ValueError("Expected N by 2 logits and N binary labels")
    if gamma_neg < 0 or gamma_pos < 0 or not torch.isfinite(torch.tensor([gamma_neg, gamma_pos])).all():
        raise ValueError("Focal exponents must be finite and nonnegative")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Labels must be 0 or 1")
    # FP32 log-softmax stays finite for extreme half-precision logits. Unlike
    # log(clamp(p, 1e-8)), its lower bound cannot underflow in FP16 autocast.
    log_probability = F.log_softmax(logits.float(), dim=-1).gather(1, labels.long().reshape(-1, 1)).squeeze(1)
    gamma = torch.where(labels == 1, float(gamma_pos), float(gamma_neg))
    values = -(1.0 - log_probability.exp()).clamp(min=0).pow(gamma) * log_probability
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError("Reduction must be none, sum, or mean")


def classification_loss(logits, labels, method):
    import torch.nn.functional as F
    if method == "ce":
        return F.cross_entropy(logits.float(), labels.long())
    if method == "custom_asl":
        return binary_asymmetric_focal_loss(logits, labels)
    raise ValueError("Unknown loss method")
