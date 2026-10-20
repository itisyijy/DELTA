from __future__ import annotations

import torch
import torch.nn as nn

from scripts.models.revin_dlinear import RevINDLinear


def _set_module_grad(module: nn.Module | nn.ModuleList, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(enabled)


def configure_tta_model(model: RevINDLinear, update_scope: str) -> None:
    scope = update_scope.lower()
    if scope not in {"all", "norm", "trend", "season"}:
        raise ValueError(f"Unknown TTA update_scope: {update_scope!r}")
    for param in model.parameters():
        param.requires_grad_(False)
    if hasattr(model.revin, "affine_weight"):
        model.revin.affine_weight.requires_grad_(scope == "norm")
    if hasattr(model.revin, "affine_bias"):
        model.revin.affine_bias.requires_grad_(scope == "norm")
    if scope == "all":
        _set_module_grad(model.dlinear, True)
    if scope == "trend":
        _set_module_grad(model.linear_trend, True)
    if scope == "season":
        _set_module_grad(model.linear_seasonal, True)


def get_tta_parameters(model: RevINDLinear) -> list[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def compute_drift_score(
    x_recent: torch.Tensor,
    mu_hist: float,
    sigma_hist: float,
) -> float:
    sigma = max(float(sigma_hist), 1e-8)
    recent = x_recent.float()
    mean_shift = (recent.mean().item() - mu_hist) / sigma
    std_shift = (recent.std(unbiased=False).item() - sigma_hist) / sigma
    return max(abs(mean_shift), abs(std_shift))


def compute_drift_score_zspace(x_recent: torch.Tensor) -> float:
    """Unit-consistent drift score (KCI 개정 B1).

    ``x_recent`` is already standardized with the client's train-split scaler, so
    the reference statistics in that space are mu = 0, sigma = 1.  ``compute_drift_score``
    instead subtracts the *raw-unit* scaler parameters from an already z-scored
    window, which makes the score dominated by the per-client constant mu_raw/sigma_raw.
    This function keeps both operands in the same space; it is opt-in via
    ``TTAConfig.drift_score_space = "zscore"`` and never changes the default path.
    """
    recent = x_recent.float()
    mean_shift = recent.mean().item()
    std_shift = recent.std(unbiased=False).item() - 1.0
    return max(abs(mean_shift), abs(std_shift))


def should_adapt(
    x_recent: torch.Tensor,
    mu_hist: float,
    sigma_hist: float,
    threshold: float,
    space: str = "raw",
) -> bool:
    if threshold <= 0:
        return True
    if space == "zscore":
        score = compute_drift_score_zspace(x_recent)
    elif space == "raw":
        score = compute_drift_score(x_recent, mu_hist, sigma_hist)
    else:
        raise ValueError(f"Unknown drift_score_space: {space!r}")
    return score >= threshold
