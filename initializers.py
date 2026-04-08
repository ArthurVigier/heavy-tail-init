"""
Stratégies d'initialisation pour l'expérience 5 : He vs. queues lourdes.

Chaque initialiseur prend un nn.Module et ré-initialise ses poids Linear/Conv2d.
Toutes les stratégies sont calibrées pour avoir Var(W_ij) ≈ 2/n_in (parité He).
"""

import math
from enum import Enum
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import levy_stable


class InitStrategy(str, Enum):
    """Stratégies d'initialisation disponibles."""

    HE_NORMAL = "he_normal"
    ALPHA_STABLE_15 = "alpha_stable_1.5"
    ALPHA_STABLE_18 = "alpha_stable_1.8"
    SIGNED_LOGNORMAL = "signed_lognormal"
    MIXTURE_INVGAMMA = "mixture_invgamma"


# --- Helpers ------------------------------------------------------------------


def _he_std(fan_in: int) -> float:
    """Écart-type He/Kaiming : sqrt(2 / fan_in)."""
    return math.sqrt(2.0 / fan_in)


def _get_fan_in(tensor: torch.Tensor) -> int:
    """Retourne le fan_in d'un poids Linear ou Conv2d."""
    if tensor.dim() == 2:  # Linear
        return tensor.shape[1]
    elif tensor.dim() == 4:  # Conv2d
        receptive = tensor.shape[2] * tensor.shape[3]
        return tensor.shape[1] * receptive
    else:
        return tensor.shape[-1]


# --- Initialiseurs -----------------------------------------------------------


def init_he_normal(tensor: torch.Tensor) -> None:
    """Initialisation He/Kaiming standard : W ~ N(0, 2/n_in)."""
    fan_in = _get_fan_in(tensor)
    std = _he_std(fan_in)
    nn.init.normal_(tensor, mean=0.0, std=std)


def init_alpha_stable(tensor: torch.Tensor, alpha: float = 1.5) -> None:
    """
    Initialisation α-stable tronquée.

    W_ij ~ S_α(σ) tronqué à [-5σ_He, 5σ_He].
    Le paramètre d'échelle σ est calibré pour que la distribution tronquée
    ait une variance effective ≈ 2/n_in.

    Pour α < 2, S_α a variance infinie, donc on ajuste σ empiriquement
    via un échantillon pilote de 100K valeurs.
    """
    fan_in = _get_fan_in(tensor)
    target_var = 2.0 / fan_in
    clip_bound = 5.0 * _he_std(fan_in)

    # Échantillon pilote pour calibrer σ
    pilot_size = 100_000
    # sigma initial : approximation grossière
    sigma_init = _he_std(fan_in) * 0.5

    # Recherche dichotomique du bon σ
    sigma_lo, sigma_hi = sigma_init * 0.01, sigma_init * 10.0
    for _ in range(30):  # bisection
        sigma_mid = (sigma_lo + sigma_hi) / 2.0
        samples = levy_stable.rvs(alpha, beta=0, loc=0, scale=sigma_mid, size=pilot_size)
        samples = np.clip(samples, -clip_bound, clip_bound)
        var_emp = np.var(samples)
        if var_emp < target_var:
            sigma_lo = sigma_mid
        else:
            sigma_hi = sigma_mid

    # Génération finale
    n = tensor.numel()
    samples = levy_stable.rvs(alpha, beta=0, loc=0, scale=sigma_mid, size=n)
    samples = np.clip(samples, -clip_bound, clip_bound)

    tensor.data.copy_(torch.from_numpy(samples.astype(np.float32)).reshape(tensor.shape))


def init_signed_lognormal(tensor: torch.Tensor) -> None:
    """
    Initialisation log-normale signée (bio-inspirée).

    W_ij = s_ij * X_ij, où s_ij ~ Uniform({-1, +1}), X_ij ~ LogNormal(μ, σ²).
    Calibré pour Var(W_ij) = 2/n_in.

    La distribution synaptique biologique est approximativement log-normale
    (Buzsáki & Mizuseki, 2014). Cette init produit une distribution
    leptokurtique (queues plus lourdes que la gaussienne) tout en conservant
    une variance finie.

    Calibration : Var(W) = E[X²] = exp(2μ + 2σ²) - exp(2μ + σ²) + exp(2μ + σ²)
                         = exp(2μ + 2σ²)
    On fixe σ_ln = 0.5 (contrôle la lourdeur des queues), puis :
      exp(2μ + 2σ²) = 2/n_in  ⟹  μ = (ln(2/n_in) - 2σ²) / 2
    """
    fan_in = _get_fan_in(tensor)
    target_var = 2.0 / fan_in

    sigma_ln = 0.5  # queue modérément lourde
    # E[X²] = exp(2μ + 2σ²), on veut E[X²] = target_var
    mu_ln = (math.log(target_var) - 2 * sigma_ln**2) / 2.0

    n = tensor.numel()
    magnitudes = np.random.lognormal(mean=mu_ln, sigma=sigma_ln, size=n)
    signs = np.random.choice([-1.0, 1.0], size=n)
    samples = signs * magnitudes

    tensor.data.copy_(torch.from_numpy(samples.astype(np.float32)).reshape(tensor.shape))


def init_mixture_invgamma(
    tensor: torch.Tensor, ig_alpha: float = 3.0, per_neuron: bool = True
) -> None:
    """
    Initialisation par mélange d'échelles (Inverse-Gamma).

    W_ij | λ_i ~ N(0, λ_i),  λ_i ~ InvGamma(α_ig, β_ig)
    Calibré pour E[λ_i] = 2/n_in.

    E[InvGamma(α, β)] = β/(α-1) pour α > 1.
    On fixe α_ig et résout β_ig = (α_ig - 1) * 2/n_in.

    Le résultat est un mélange de gaussiennes à variances aléatoires →
    distribution marginale Student-t (pour α_ig = ν/2, β_ig = ν*σ²/2,
    on obtient exactement une Student-t à ν degrés de liberté).
    """
    fan_in = _get_fan_in(tensor)
    target_mean_var = 2.0 / fan_in

    if ig_alpha <= 1.0:
        raise ValueError("ig_alpha doit être > 1 pour que E[λ] existe")

    beta_ig = (ig_alpha - 1.0) * target_mean_var

    if tensor.dim() == 2:  # Linear(out, in)
        n_out, n_in = tensor.shape
        if per_neuron:
            # Un λ par neurone de sortie
            lambdas = np.random.gamma(
                shape=ig_alpha, scale=1.0 / beta_ig, size=n_out
            )
            # Inverser (InvGamma = 1/Gamma)
            lambdas = 1.0 / lambdas
            # Générer W : chaque ligne a sa propre variance
            samples = np.zeros((n_out, n_in), dtype=np.float32)
            for i in range(n_out):
                samples[i] = np.random.normal(0, math.sqrt(lambdas[i]), size=n_in)
        else:
            # Un λ par poids
            lambdas = 1.0 / np.random.gamma(
                shape=ig_alpha, scale=1.0 / beta_ig, size=tensor.numel()
            )
            samples = np.random.normal(0, 1, size=tensor.numel()) * np.sqrt(lambdas)
            samples = samples.astype(np.float32).reshape(tensor.shape)
    else:
        # Fallback : un λ par poids
        n = tensor.numel()
        lambdas = 1.0 / np.random.gamma(shape=ig_alpha, scale=1.0 / beta_ig, size=n)
        samples = (np.random.normal(0, 1, size=n) * np.sqrt(lambdas)).astype(np.float32)
        samples = samples.reshape(tensor.shape)

    tensor.data.copy_(torch.from_numpy(samples))


# --- Dispatch -----------------------------------------------------------------


def get_initializer(strategy: InitStrategy) -> Callable[[torch.Tensor], None]:
    """Retourne la fonction d'initialisation pour une stratégie donnée."""
    dispatch: dict[InitStrategy, Callable[[torch.Tensor], None]] = {
        InitStrategy.HE_NORMAL: init_he_normal,
        InitStrategy.ALPHA_STABLE_15: lambda t: init_alpha_stable(t, alpha=1.5),
        InitStrategy.ALPHA_STABLE_18: lambda t: init_alpha_stable(t, alpha=1.8),
        InitStrategy.SIGNED_LOGNORMAL: init_signed_lognormal,
        InitStrategy.MIXTURE_INVGAMMA: init_mixture_invgamma,
    }
    return dispatch[strategy]


def apply_init(model: nn.Module, strategy: InitStrategy) -> None:
    """
    Applique une stratégie d'initialisation à tous les poids Linear et Conv2d.

    Les biais sont mis à zéro. Les LayerNorm et embeddings positionnels
    ne sont PAS touchés (ils gardent l'init par défaut de timm).
    """
    init_fn = get_initializer(strategy)
    count = 0
    for name, param in model.named_parameters():
        if param.dim() < 2:
            # Biais, gains de LayerNorm → zéro ou garder défaut
            if "bias" in name:
                nn.init.zeros_(param)
            continue
        # Ne pas toucher aux embeddings de position/classe
        if "pos_embed" in name or "cls_token" in name:
            continue
        init_fn(param.data)
        count += 1

    print(f"  [init] {strategy.value}: {count} tenseurs de poids ré-initialisés")


# --- Diagnostics --------------------------------------------------------------


def weight_stats(model: nn.Module) -> dict[str, float]:
    """Calcule les statistiques des poids (kurtosis, moments) pour diagnostic."""
    all_weights: list[float] = []
    for name, param in model.named_parameters():
        if param.dim() >= 2 and "pos_embed" not in name and "cls_token" not in name:
            all_weights.extend(param.data.cpu().numpy().ravel().tolist())

    w = np.array(all_weights)
    stats = {
        "mean": float(np.mean(w)),
        "std": float(np.std(w)),
        "kurtosis": float(np.mean((w - np.mean(w)) ** 4) / np.std(w) ** 4 - 3),
        "min": float(np.min(w)),
        "max": float(np.max(w)),
        "abs_p99": float(np.percentile(np.abs(w), 99)),
    }
    return stats
