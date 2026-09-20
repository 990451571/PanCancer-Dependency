"""GPU estimators used by the frozen advanced dependency benchmark."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def standardize_expression(train_values: np.ndarray, held_values: np.ndarray, dtype=torch.float32):
    """Training-only mean/scale standardization with constant columns removed."""
    x = torch.as_tensor(train_values, dtype=torch.float64, device="cuda")
    hx = torch.as_tensor(held_values, dtype=torch.float64, device="cuda")
    mean = x.mean(0)
    centered = x - mean
    sd = torch.sqrt((centered * centered).mean(0))
    keep = sd > 1e-8
    x = (centered[:, keep] / sd[keep]).to(dtype)
    hx = ((hx[:, keep] - mean[keep]) / sd[keep]).to(dtype)
    return x, hx, keep.cpu().numpy()


def expression_kernels(train_values: np.ndarray, held_values: np.ndarray):
    x, hx, keep = standardize_expression(train_values, held_values, torch.float64)
    return x @ x.T, hx @ x.T, x, hx, keep


def _soft_threshold(value: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.sign(value) * torch.clamp(torch.abs(value) - threshold, min=0.0)


def elastic_net_path(
    x: torch.Tensor,
    y: np.ndarray,
    hx: torch.Tensor,
    alphas: tuple[float, ...],
    l1_ratio: float = 0.5,
    maximum_iterations: int = 1000,
    tolerance: float = 1e-5,
    audit_callback=None,
):
    """Historical proximal solver with the sklearn-scaled per-target objective.

    Each target has its own coefficient vector. The vectorized implementation
    shares matrix multiplications but does not couple target coefficients.
    The historical restart cancels acceleration; retained for reproducibility.
    An optional read-only callback supports numerical audits without altering fits.
    """
    if x.dtype != torch.float32 or hx.dtype != torch.float32:
        raise ValueError("Elastic Net expects float32 CUDA expression matrices")
    observed_np = np.isfinite(y)
    count_np = observed_np.sum(0)
    if (count_np < 2).any():
        raise ValueError("Elastic Net target with fewer than two observations")
    outputs = {alpha: np.full((len(hx), y.shape[1]), np.nan, dtype=np.float32) for alpha in alphas}
    group_diagnostics = defaultdict(list)
    _, groups = np.unique(np.packbits(observed_np.T, axis=1), axis=0, return_inverse=True)
    for group in np.unique(groups):
        columns = np.flatnonzero(groups == group)
        rows = np.flatnonzero(observed_np[:, columns[0]])
        gx = x[torch.as_tensor(rows, device="cuda")]
        group_feature_mean = gx.mean(0)
        gx = gx - group_feature_mean
        ghx = hx - group_feature_mean
        gy_np = y[np.ix_(rows, columns)]
        mean_np = gy_np.mean(0)
        outcomes = torch.as_tensor(gy_np - mean_np, dtype=torch.float32, device="cuda")
        spectral = float(torch.linalg.eigvalsh(gx @ gx.T)[-1].item())
        weight = torch.zeros((x.shape[1], len(columns)), dtype=torch.float32, device="cuda")
        for alpha in sorted(alphas, reverse=True):
            current = weight.clone()
            accelerated = current.clone()
            momentum = 1.0
            lipschitz = spectral / len(rows) + alpha * (1.0 - l1_ratio)
            step = 1.0 / lipschitz
            converged = False
            relative = math.inf
            for iteration in range(1, maximum_iterations + 1):
                residual = gx @ accelerated - outcomes
                gradient = gx.T @ residual / len(rows)
                gradient.add_(accelerated, alpha=alpha * (1.0 - l1_ratio))
                updated = _soft_threshold(accelerated - step * gradient, step * alpha * l1_ratio)
                next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
                candidate = updated + ((momentum - 1.0) / next_momentum) * (updated - current)
                # Adaptive restart suppresses FISTA oscillation without changing the objective.
                if torch.sum((updated - current) * (candidate - updated)) > 0:
                    accelerated, next_momentum = updated, 1.0
                else:
                    accelerated = candidate
                if iteration % 10 == 0 or iteration == maximum_iterations:
                    denominator = max(float(torch.linalg.vector_norm(updated).item()), 1e-12)
                    relative = float(torch.linalg.vector_norm(updated - current).item()) / denominator
                    if relative <= tolerance:
                        converged = True
                        current = updated
                        break
                current, momentum = updated, next_momentum
            weight = current
            prediction = ghx @ weight + torch.as_tensor(mean_np, dtype=torch.float32, device="cuda")
            outputs[alpha][:, columns] = prediction.cpu().numpy()
            if audit_callback is not None:
                audit_callback(alpha, columns, gx, outcomes, weight, ghx, mean_np)
            group_diagnostics[alpha].append({
                "iterations": iteration,
                "relative_change": relative,
                "converged": converged,
                "nonzero_n": int((weight != 0).sum().item()),
                "coefficient_n": weight.numel(),
                "lipschitz": lipschitz,
                "target_n": len(columns),
                "observed_model_n": len(rows),
            })
    diagnostics = {}
    for alpha, records in group_diagnostics.items():
        diagnostics[alpha] = {
            "iterations": max(record["iterations"] for record in records),
            "relative_change": max(record["relative_change"] for record in records),
            "converged": all(record["converged"] for record in records),
            "nonzero_fraction": sum(record["nonzero_n"] for record in records)
                                / sum(record["coefficient_n"] for record in records),
            "lipschitz": max(record["lipschitz"] for record in records),
            "mask_group_n": len(records),
            "minimum_observed_model_n": min(record["observed_model_n"] for record in records),
        }
    return outputs, diagnostics


def reduced_rank_ridge_predictions(
    kernel: torch.Tensor,
    held_kernel: torch.Tensor,
    residual: np.ndarray,
    ranks: tuple[int, ...],
    alphas: tuple[float, ...],
):
    """Multi-task ridge followed by a training-only target-space SVD projection."""
    if kernel.dtype != torch.float64:
        raise ValueError("Reduced-rank ridge expects float64 kernels")
    observed = np.isfinite(residual)
    counts = observed.sum(0)
    means = np.divide(np.nansum(residual, axis=0), counts,
                      out=np.zeros(residual.shape[1]), where=counts > 0)
    filled = np.where(observed, residual - means, 0.0)
    y = torch.as_tensor(filled, dtype=torch.float64, device="cuda")
    eye = torch.eye(kernel.shape[0], dtype=torch.float64, device="cuda")
    outputs, diagnostics = {}, {}
    for alpha in alphas:
        dual = torch.linalg.solve(kernel + alpha * eye, y)
        fitted = kernel @ dual
        held = held_kernel @ dual
        _, singular, vh = torch.linalg.svd(fitted, full_matrices=False)
        total = float((singular * singular).sum().item())
        for requested_rank in ranks:
            rank = min(requested_rank, vh.shape[0])
            basis = vh[:rank]
            prediction = (held @ basis.T) @ basis
            prediction.add_(torch.as_tensor(means, dtype=torch.float64, device="cuda"))
            outputs[(requested_rank, alpha)] = prediction.cpu().numpy()
            diagnostics[(requested_rank, alpha)] = {
                "effective_rank": rank,
                "fitted_variance_fraction": float((singular[:rank] ** 2).sum().item() / total) if total else 0.0,
                "training_missing_fraction": float((~observed).mean()),
            }
    return outputs, diagnostics


class ExpressionAutoencoder(nn.Module):
    def __init__(self, input_n: int = 6016):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_n, 500), nn.ReLU(),
            nn.Linear(500, 200), nn.ReLU(),
            nn.Linear(200, 50), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(50, 200), nn.ReLU(),
            nn.Linear(200, 500), nn.ReLU(),
            nn.Linear(500, input_n), nn.ReLU(),
        )

    def forward(self, values):
        return self.decoder(self.encoder(values))


class ExpDeepDEPAdapted(nn.Module):
    def __init__(self, expression_encoder: nn.Module, fingerprint_n: int = 3115):
        super().__init__()
        self.expression_encoder = copy.deepcopy(expression_encoder)
        self.fingerprint_encoder = nn.Sequential(
            nn.Linear(fingerprint_n, 1000), nn.ReLU(),
            nn.Linear(1000, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(100, 250), nn.ReLU(),
            nn.Linear(250, 250), nn.ReLU(),
            nn.Linear(250, 1),
        )

    def encoded(self, expression, fingerprints):
        return self.expression_encoder(expression), self.fingerprint_encoder(fingerprints)

    def cartesian(self, expression, fingerprints, gene_chunk: int = 256):
        sample_embedding, target_embedding = self.encoded(expression, fingerprints)
        pieces = []
        for start in range(0, len(target_embedding), gene_chunk):
            target = target_embedding[start:start + gene_chunk]
            sample = sample_embedding[:, None, :].expand(-1, len(target), -1)
            gene = target[None, :, :].expand(len(sample_embedding), -1, -1)
            pair = torch.cat((sample, gene), dim=2)
            pieces.append(self.predictor(pair).squeeze(2))
        return torch.cat(pieces, dim=1)


def initialize_he(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)


@dataclass
class DeepTrainingResult:
    prediction: np.ndarray
    epoch_n: int
    validation_loss: float
    training_loss: float


def pretrain_expression_autoencoder(
    values: np.ndarray,
    seed: int,
    epochs: int = 100,
    batch_size: int = 64,
    report_every: int = 10,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = ExpressionAutoencoder(values.shape[1]).cuda()
    initialize_he(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tensor = torch.as_tensor(values, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    model.train()
    for epoch in range(epochs):
        total, count = 0.0, 0
        for indices in torch.randperm(len(tensor), generator=generator).split(batch_size):
            batch = tensor[indices].cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = torch.mean((prediction - batch) ** 2)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch)
            count += len(batch)
        losses.append(total / count)
        if epoch == 0 or (epoch + 1) % report_every == 0 or epoch + 1 == epochs:
            print(f"  自编码器 epoch {epoch + 1}/{epochs}｜重建MSE {losses[-1]:.6f}", flush=True)
    return model.encoder.cpu(), losses


def _deepdep_loss(model, expression, fingerprints, outcomes, observed, batch_models, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_sse = torch.zeros((), dtype=torch.float32, device="cuda")
    total_n = 0
    order = torch.randperm(len(expression), device="cuda") if train else torch.arange(len(expression), device="cuda")
    for indices in order.split(batch_models):
        if train:
            optimizer.zero_grad(set_to_none=True)
        prediction = model.cartesian(expression[indices], fingerprints)
        mask = observed[indices]
        error = prediction[mask] - outcomes[indices][mask]
        loss = torch.mean(error * error)
        if train:
            loss.backward()
            optimizer.step()
        total_sse += (error.detach() * error.detach()).sum()
        total_n += int(mask.sum().item())
    return float((total_sse / total_n).item())


def train_exp_deepdep(
    train_expression: np.ndarray,
    train_outcomes: np.ndarray,
    held_expression: np.ndarray,
    fingerprints: np.ndarray,
    pretrained_encoder: nn.Module,
    seed: int,
    epochs: int,
    validation_expression: np.ndarray | None = None,
    validation_outcomes: np.ndarray | None = None,
    patience: int = 3,
    batch_models: int = 32,
):
    """Train fixed-architecture Exp-DeepDEP; optional validation selects epoch count."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = ExpDeepDEPAdapted(pretrained_encoder).cuda()
    initialize_he(model.fingerprint_encoder)
    initialize_he(model.predictor)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.as_tensor(train_expression, dtype=torch.float32, device="cuda")
    y = torch.as_tensor(np.nan_to_num(train_outcomes, nan=0.0), dtype=torch.float32, device="cuda")
    mask = torch.as_tensor(np.isfinite(train_outcomes), dtype=torch.bool, device="cuda")
    fp = torch.as_tensor(fingerprints, dtype=torch.float32, device="cuda")
    if validation_expression is not None:
        vx = torch.as_tensor(validation_expression, dtype=torch.float32, device="cuda")
        vy = torch.as_tensor(np.nan_to_num(validation_outcomes, nan=0.0), dtype=torch.float32, device="cuda")
        vm = torch.as_tensor(np.isfinite(validation_outcomes), dtype=torch.bool, device="cuda")
    best_state, best_loss, stale, best_epoch = None, math.inf, 0, epochs
    train_loss = math.nan
    for epoch in range(1, epochs + 1):
        train_loss = _deepdep_loss(model, x, fp, y, mask, batch_models, optimizer)
        if validation_expression is None:
            continue
        with torch.no_grad():
            validation_loss = _deepdep_loss(model, vx, fp, vy, vm, batch_models, None)
        if validation_loss < best_loss - 1e-7:
            best_loss, stale, best_epoch = validation_loss, 0, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    hx = torch.as_tensor(held_expression, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        prediction = model.cartesian(hx, fp).cpu().numpy()
    return DeepTrainingResult(prediction, best_epoch, best_loss, train_loss)
