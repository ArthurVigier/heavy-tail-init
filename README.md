# Heavy-Tail Init: Does biological initialization beat Glorot/He?

**Experiment 5** from the research report *"L'hypothèse gaussienne en deep learning"*.

## Thesis

SGD naturally pushes weights toward heavy-tailed distributions during training (Martin & Mahoney, HT-SR program). Biological synaptic distributions are log-normal. Yet we initialize with Gaussian (He/Glorot). **What if we start where SGD wants to end up?**

## What this does

Trains **ViT-S/16 on CIFAR-100** with 5 initialization strategies, everything else identical:

| Strategy | Distribution | Kurtosis | Bio-inspired? |
|---|---|---|---|
| `he_normal` | $W \sim \mathcal{N}(0, 2/n)$ | 0 (baseline) | No |
| `alpha_stable_1.5` | $W \sim S_{1.5}(\sigma)$ truncated | ~10-50 | Partially |
| `alpha_stable_1.8` | $W \sim S_{1.8}(\sigma)$ truncated | ~3-10 | Partially |
| `signed_lognormal` | $W = \pm \text{LogNormal}(\mu, \sigma^2)$ | ~5-15 | **Yes** |
| `mixture_invgamma` | $W \mid \lambda \sim \mathcal{N}(0,\lambda)$, $\lambda \sim \text{InvGamma}$ | ~3-8 | Partially |

All calibrated to match He variance: $\text{Var}(W_{ij}) = 2/n_{in}$.

## Quick start

```bash
# Clone
git clone <repo-url> && cd experiment_init

# Install
pip install -r requirements.txt

# Dry run (2 epochs, 2 inits, ~5 min on any GPU)
python run_vit_cifar100.py --dry-run

# Full experiment (300 epochs, 5 inits × 5 seeds = 25 runs)
python run_vit_cifar100.py

# Custom
python run_vit_cifar100.py --epochs 100 --seeds 3 --inits he_normal signed_lognormal alpha_stable_1.8
```

## GPU rental setup (Lambda/Vast.ai/RunPod)

```bash
# 1. SSH into your instance
ssh user@<instance-ip>

# 2. Clone and setup
git clone <repo-url> && cd experiment_init
pip install -r requirements.txt

# 3. Full run in tmux (survives SSH disconnect)
tmux new -s experiment
python run_vit_cifar100.py 2>&1 | tee run.log

# 4. Monitor from another tmux pane
watch -n 30 'tail -5 results/log_*.jsonl | python -m json.tool'
```

### Recommended GPU

- **A100 40GB**: ~5 GPU-days for full experiment (25 runs × 300 epochs)
- **A10 24GB**: works, ~8 GPU-days
- **RTX 4090 24GB**: works, ~6 GPU-days
- Single run (1 init, 1 seed, 300 epochs): ~5h on A100

### Cost estimate

| Provider | GPU | $/hr | Full experiment | Single run |
|---|---|---|---|---|
| Lambda | A100 | ~$1.10 | ~$130 | ~$5.50 |
| Vast.ai | A100 | ~$0.80 | ~$96 | ~$4.00 |
| RunPod | A100 | ~$0.74 | ~$89 | ~$3.70 |

**Tip**: run He + signed_lognormal first (2 runs, ~$8) to see if the signal is there before committing to the full matrix.

## Output

### During training
```
🧪 signed_lognormal | seed=42 | 300 epochs
  Init kurtosis: 8.42 | std: 0.0234 | |w|_p99: 0.0891
  Epoch  50/300 | test_acc=62.3% | best=62.3%
  Epoch 100/300 | test_acc=71.8% | best=71.8%
  ...
```

### Final comparison table
```
Stratégie              | Best Acc |    ±σ | Diverg. | Kurt init | Kurt final
---------------------------------------------------------------------------
he_normal              |  78.42% |  0.31 |   0/ 5  |       0.0 |       12.3
signed_lognormal       |  79.87% |  0.28 |   0/ 5  |       8.4 |       11.8
alpha_stable_1.8       |  79.12% |  0.45 |   0/ 5  |       5.2 |       12.1
...
```

### Files
- `results/log_<timestamp>.jsonl` — per-run logs (JSON lines)
- `results/results_<timestamp>.json` — full results with training curves

## Key metrics to watch

1. **Best accuracy** — does heavy-tail init beat He?
2. **Convergence speed** — epochs to reach 70% accuracy
3. **Final kurtosis** — do all inits converge to the same weight distribution?
4. **Divergence rate** — stability of heavy-tail inits
5. **Init kurtosis → final kurtosis gap** — does He "waste" epochs building heavy tails?

## Kill criteria

- If >50% of α-stable runs diverge → heavy-tail init needs architectural changes (adapted normalization)
- If He dominates on all metrics → Gaussian init is sufficient (negative result, still publishable)

## Architecture

```
experiment_init/
├── run_vit_cifar100.py    # Main experiment (train + eval + compare)
├── initializers.py        # 5 init strategies + calibration + diagnostics
├── requirements.txt
├── .gitignore
└── results/               # Generated at runtime
```

## References

- Martin & Mahoney (2021) "Implicit Self-Regularization in DNN" — HT-SR program
- Buzsáki & Mizuseki (2014) "The log-dynamic brain" — biological log-normal distributions
- Lynn, Holmes & Palmer (Nature Physics 2024) — heavy-tailed neuronal connectivity
- He et al. (2015) "Delving Deep into Rectifiers" — He/Kaiming initialization
