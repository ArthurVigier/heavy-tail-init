#!/usr/bin/env python3
"""
Expérience 5 — Initialisation α-stable vs. He pour ViT-S/16 sur CIFAR-100.

Compare 5 stratégies d'initialisation à architecture et hyperparamètres identiques.
Mesure : accuracy top-1, vitesse de convergence, kurtosis des poids post-entraînement.

Usage:
  # Dry-run (1 epoch, 2 inits, 1 seed)
  python run_vit_cifar100.py --dry-run

  # Full run (300 epochs, 5 inits, 5 seeds)
  python run_vit_cifar100.py

  # Spécifique
  python run_vit_cifar100.py --epochs 100 --seeds 3 --inits he_normal signed_lognormal
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import timm

from initializers import InitStrategy, apply_init, weight_stats

# --- Config -------------------------------------------------------------------

RESULTS_DIR = Path("results")
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Hyperparamètres (identiques pour toutes les inits)
_num_workers = min(os.cpu_count() or 4, 8)
DEFAULT_CONFIG = {
    "model": "vit_small_patch16_224",
    "dataset": "CIFAR-100",
    "num_classes": 100,
    "img_size": 224,
    "epochs": 300,
    "batch_size": 512 if torch.cuda.is_available() else 128,
    "lr": 1e-3,
    "weight_decay": 0.05,
    "warmup_epochs": 10,
    "label_smoothing": 0.1,
    "mixup_alpha": 0.8,
    "cutmix_alpha": 1.0,
    "drop_path": 0.1,
    "num_workers": _num_workers,
}


# --- Data ---------------------------------------------------------------------


class CachedCIFAR100(Dataset):
    """
    CIFAR-100 avec images pré-resizées et cachées en RAM comme tenseurs.

    Le bottleneck principal sur GPU loué (peu de vCPUs) est le resize
    32→224 par image par epoch. On le fait UNE FOIS au démarrage,
    et on cache les tenseurs normalisés en RAM (~7.5 GB pour 224×224).
    Les augmentations aléatoires restent dynamiques.
    """

    def __init__(
        self,
        root: str,
        train: bool,
        img_size: int,
        download: bool = True,
    ) -> None:
        self.dataset = datasets.CIFAR100(root=root, train=train, download=download)
        self.targets = self.dataset.targets
        self.train = train
        self.img_size = img_size

        # Pré-resize seulement, stocké en uint8 (1 byte/pixel vs 4)
        resize = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),  # [0,1] float32 temporaire
        ])

        split = "train" if train else "test"
        n = len(self.dataset)
        print(f"  ⏳ Caching {n} {split} images at {img_size}×{img_size} (uint8)...", end=" ", flush=True)
        t0 = time.time()

        # Stocker en uint8 : 7.5 GB train au lieu de 30 GB
        self.images = torch.empty(n, 3, img_size, img_size, dtype=torch.uint8)
        for i in range(n):
            img, _ = self.dataset[i]
            self.images[i] = (resize(img) * 255).to(torch.uint8)

        # Libérer le dataset original
        del self.dataset

        elapsed = time.time() - t0
        mem_gb = self.images.nelement() * self.images.element_size() / 1e9
        print(f"done in {elapsed:.0f}s ({mem_gb:.1f} GB)")

        # Normalisation appliquée dynamiquement (rapide sur tenseur)
        self.normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        )

        # Augmentations dynamiques (appliquées à chaque __getitem__)
        if train:
            self.augment = transforms.Compose([
                transforms.RandomCrop(img_size, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.RandomErasing(p=0.25),
            ])
        else:
            self.augment = None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = self.images[idx].float() / 255.0  # uint8 → float32 [0,1]
        img = self.normalize(img)
        if self.augment is not None:
            img = self.augment(img)
        return img, self.targets[idx]


# Cache global pour éviter de recharger entre les runs
_data_cache: dict[str, CachedCIFAR100] = {}


def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """Construit les dataloaders CIFAR-100 avec cache en RAM."""
    global _data_cache

    if "train" not in _data_cache:
        _data_cache["train"] = CachedCIFAR100(
            root="./data", train=True, img_size=config["img_size"], download=True,
        )
    if "test" not in _data_cache:
        _data_cache["test"] = CachedCIFAR100(
            root="./data", train=False, img_size=config["img_size"], download=True,
        )

    pw = config["num_workers"] > 0
    train_loader = DataLoader(
        _data_cache["train"],
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=pw,
    )
    test_loader = DataLoader(
        _data_cache["test"],
        batch_size=config["batch_size"] * 2,
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=pw,
    )
    return train_loader, test_loader


# --- Model --------------------------------------------------------------------


def build_model(config: dict, strategy: InitStrategy) -> nn.Module:
    """Construit ViT-S/16 et applique la stratégie d'initialisation."""
    model = timm.create_model(
        config["model"],
        pretrained=False,
        num_classes=config["num_classes"],
        drop_path_rate=config["drop_path"],
    )
    apply_init(model, strategy)
    return model.to(DEVICE)


# --- Training -----------------------------------------------------------------


def cosine_lr_with_warmup(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    base_lr: float,
) -> float:
    """Cosine LR schedule avec warmup linéaire."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = base_lr * 0.5 * (1 + np.cos(np.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def mixup_cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    mixup_alpha: float = 0.8,
    cutmix_alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applique MixUp ou CutMix aléatoirement (50/50)."""
    targets_onehot = torch.nn.functional.one_hot(targets, num_classes).float()

    if np.random.random() < 0.5 and mixup_alpha > 0:
        # MixUp
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        idx = torch.randperm(images.size(0), device=images.device)
        images = lam * images + (1 - lam) * images[idx]
        targets_onehot = lam * targets_onehot + (1 - lam) * targets_onehot[idx]
    elif cutmix_alpha > 0:
        # CutMix
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        idx = torch.randperm(images.size(0), device=images.device)
        _, _, h, w = images.shape
        cut_h = int(h * np.sqrt(1 - lam))
        cut_w = int(w * np.sqrt(1 - lam))
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1 = max(0, cy - cut_h // 2)
        y2 = min(h, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(w, cx + cut_w // 2)
        images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
        lam_actual = 1 - (y2 - y1) * (x2 - x1) / (h * w)
        targets_onehot = lam_actual * targets_onehot + (1 - lam_actual) * targets_onehot[idx]

    return images, targets_onehot


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> float:
    """Entraîne un epoch, retourne la loss moyenne."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])

    for images, targets in loader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        # MixUp / CutMix
        images, targets_mixed = mixup_cutmix(
            images, targets, config["num_classes"],
            config["mixup_alpha"], config["cutmix_alpha"],
        )

        logits = model(images)
        loss = (-targets_mixed * torch.nn.functional.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    """Évalue le modèle, retourne (loss, accuracy top-1)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for images, targets in loader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * targets.size(0)
        correct += (logits.argmax(dim=-1) == targets).sum().item()
        total += targets.size(0)

    return total_loss / total, correct / total


# --- Run one experiment -------------------------------------------------------


def run_single(
    strategy: InitStrategy,
    seed: int,
    config: dict,
    log_file: Path,
) -> dict:
    """Entraîne un run complet et retourne les résultats."""
    print(f"\n{'='*60}")
    print(f"🧪 {strategy.value} | seed={seed} | {config['epochs']} epochs")
    print(f"{'='*60}")

    # Reproductibilité
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Data
    train_loader, test_loader = build_dataloaders(config)

    # Model
    model = build_model(config, strategy)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Paramètres: {n_params:.1f}M | Device: {DEVICE}")

    # Stats init
    init_stats = weight_stats(model)
    print(f"  Init kurtosis: {init_stats['kurtosis']:.2f} | "
          f"std: {init_stats['std']:.4f} | "
          f"|w|_p99: {init_stats['abs_p99']:.4f}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    # Training loop
    best_acc = 0.0
    best_epoch = 0
    history: list[dict] = []
    diverged = False
    start_time = time.time()

    for epoch in range(config["epochs"]):
        lr = cosine_lr_with_warmup(
            optimizer, epoch, config["epochs"],
            config["warmup_epochs"], config["lr"],
        )
        train_loss = train_one_epoch(model, train_loader, optimizer, config)

        # Check divergence
        if not np.isfinite(train_loss):
            print(f"  💥 DIVERGENCE à epoch {epoch}")
            diverged = True
            break

        # Eval tous les 5 epochs (ou chaque epoch si dry-run)
        eval_every = 1 if config["epochs"] <= 5 else 5
        if epoch % eval_every == 0 or epoch == config["epochs"] - 1:
            test_loss, test_acc = evaluate(model, test_loader)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch

            elapsed = time.time() - start_time
            print(
                f"  Epoch {epoch:>3d}/{config['epochs']} | "
                f"lr={lr:.2e} | train_loss={train_loss:.4f} | "
                f"test_acc={test_acc*100:.1f}% | "
                f"best={best_acc*100:.1f}% (e{best_epoch}) | "
                f"{elapsed:.0f}s"
            )

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "test_loss": round(test_loss, 5),
                "test_acc": round(test_acc, 5),
                "best_acc": round(best_acc, 5),
                "lr": round(lr, 8),
            }
            history.append(epoch_record)

    total_time = time.time() - start_time

    # Stats finales des poids
    final_stats = weight_stats(model) if not diverged else {}

    result = {
        "strategy": strategy.value,
        "seed": seed,
        "best_acc": round(best_acc, 5),
        "best_epoch": best_epoch,
        "final_acc": history[-1]["test_acc"] if history else 0.0,
        "diverged": diverged,
        "total_time_s": round(total_time, 1),
        "init_kurtosis": round(init_stats["kurtosis"], 2),
        "final_kurtosis": round(final_stats.get("kurtosis", 0), 2),
        "config": config,
        "history": history,
    }

    # Log
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **{k: v for k, v in result.items() if k != "history" and k != "config"},
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return result


# --- Comparaison finale -------------------------------------------------------


def print_comparison(results: list[dict]) -> None:
    """Affiche un tableau comparatif des résultats."""
    # Grouper par stratégie
    from collections import defaultdict
    by_strategy: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_strategy[r["strategy"]].append(r)

    print(f"\n{'='*80}")
    print("📊 RÉSULTATS COMPARATIFS — ViT-S/16 sur CIFAR-100")
    print(f"{'='*80}")
    print(
        f"{'Stratégie':<22} | {'Best Acc':>8} | {'±σ':>6} | "
        f"{'Diverg.':>7} | {'Kurt init':>9} | {'Kurt final':>10} | {'Temps':>6}"
    )
    print("-" * 80)

    for strat in InitStrategy:
        runs = by_strategy.get(strat.value, [])
        if not runs:
            continue
        accs = [r["best_acc"] for r in runs if not r["diverged"]]
        diverged = sum(1 for r in runs if r["diverged"])
        times = [r["total_time_s"] for r in runs]
        init_k = [r["init_kurtosis"] for r in runs]
        final_k = [r["final_kurtosis"] for r in runs if not r["diverged"]]

        if accs:
            acc_mean = np.mean(accs) * 100
            acc_std = np.std(accs) * 100
            kurt_i = np.mean(init_k)
            kurt_f = np.mean(final_k) if final_k else float("nan")
            avg_time = np.mean(times) / 60  # minutes
            print(
                f"{strat.value:<22} | {acc_mean:>7.2f}% | {acc_std:>5.2f} | "
                f"{diverged:>3d}/{len(runs):>2d}  | {kurt_i:>9.1f} | "
                f"{kurt_f:>10.1f} | {avg_time:>5.0f}m"
            )
        else:
            print(f"{strat.value:<22} |   TOUTES DIVERGÉES ({len(runs)} runs)")

    print("-" * 80)


# --- Main ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expérience 5 : Init α-stable vs He pour ViT-S/16 CIFAR-100"
    )
    parser.add_argument("--dry-run", action="store_true", help="Quick test (2 epochs, 2 inits, 1 seed)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None, help="Nombre de seeds")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--inits",
        nargs="+",
        default=None,
        help="Stratégies à tester (e.g., he_normal signed_lognormal)",
    )
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()

    if args.dry_run:
        config["epochs"] = 2
        config["num_workers"] = 0
        seeds = [42]
        strategies = [InitStrategy.HE_NORMAL, InitStrategy.SIGNED_LOGNORMAL]
        print("🏃 DRY RUN — 2 epochs, 2 inits, 1 seed")
    else:
        seeds = list(range(42, 42 + (args.seeds or 5)))
        strategies = (
            [InitStrategy(s) for s in args.inits]
            if args.inits
            else list(InitStrategy)
        )

    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["batch_size"] = args.batch_size

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = RESULTS_DIR / f"log_{timestamp}.jsonl"
    results_file = RESULTS_DIR / f"results_{timestamp}.json"

    print(f"📁 Résultats: {results_file}")
    print(f"📊 Log: {log_file}")
    print(f"🖥️  Device: {DEVICE}")
    print(f"📐 Config: {config['epochs']} epochs, batch_size={config['batch_size']}")
    print(f"🎯 Inits: {[s.value for s in strategies]}")
    print(f"🎲 Seeds: {seeds}")

    all_results: list[dict] = []
    for strategy in strategies:
        for seed in seeds:
            result = run_single(strategy, seed, config, log_file)
            all_results.append(result)

    # Sauvegarder tous les résultats
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print_comparison(all_results)

    print(f"\n✅ Terminé ! Résultats sauvegardés dans {results_file}")


if __name__ == "__main__":
    main()
