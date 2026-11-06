"""Head importance (Michel et al. 2019) for each M0 seed, plus which heads to prune for M25 / M50.

    python src/head_importance.py              # all seeds in configs/config.yaml
    python src/head_importance.py --seeds 42   # one seed
    python src/head_importance.py --force      # recompute finished seeds

Importance of a head = mean over training examples of |dLoss / d(xi)|, where xi is a per-head mask
variable (value 1) that scales the head's attention output. Scores are L2-normalised within each
layer so that heads can be ranked globally across layers (as in Michel et al.).

Output per seed: results/head_importance/M0_seed{seed}.json (small, safe to commit).
"""
import argparse
import json

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer

from common import (ROOT, get_device, load_checkpoint, load_config, load_splits, make_loader,
                    set_fp32_strict)

OUT_DIR = ROOT / "results" / "head_importance"


def head_importance(model, loader, device):
    """Mean |dLoss/dxi| per head over the loader, as a (layers, heads) tensor. Uses batch size 1."""
    model.eval()                                   # no dropout
    for p in model.parameters():
        p.requires_grad_(False)                    # only the head mask needs gradients
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    head_mask = torch.ones(n_layers, n_heads, device=device, requires_grad=True)
    total = torch.zeros(n_layers, n_heads, device=device)
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch, head_mask=head_mask).loss
        (grad,) = torch.autograd.grad(loss, head_mask)
        total += grad.abs()                        # absolute value per example, as in the definition
        n += batch["labels"].size(0)
    return (total / n).detach()


def normalize_per_layer(imp):
    return imp / (imp.norm(p=2, dim=1, keepdim=True) + 1e-20)


def select_heads_to_prune(norm_imp, n_prune, min_per_layer=1):
    """Globally prune the n_prune least important heads, keeping at least min_per_layer per layer.

    Returns {layer: [heads]} with int keys, the format model.prune_heads() expects.
    """
    n_layers, n_heads = norm_imp.shape
    order = sorted((float(norm_imp[l, h]), l, h) for l in range(n_layers) for h in range(n_heads))
    remaining = [n_heads] * n_layers
    pruned, count = {}, 0
    for _, l, h in order:
        if count == n_prune:
            break
        if remaining[l] <= min_per_layer:
            continue
        pruned.setdefault(l, []).append(h)
        remaining[l] -= 1
        count += 1
    assert count == n_prune, f"could only select {count} of {n_prune} heads"
    return {l: sorted(hs) for l, hs in sorted(pruned.items())}


def run_seed(seed, cfg, device, force):
    out_path = OUT_DIR / f"M0_seed{seed}.json"
    if out_path.exists() and not force:
        print(f"[seed {seed}] already done ({out_path}), skipping. Use --force to redo.")
        return
    ckpt = ROOT / "checkpoints" / "M0" / f"seed{seed}"
    model = load_checkpoint(ckpt, device)
    tok = AutoTokenizer.from_pretrained(ckpt)
    train = load_splits(cfg, tok)["train"]
    loader = make_loader(train, tok, batch_size=1)

    # Guard: an all-ones head mask must not change the output, and gradients must actually flow.
    first = {k: v.to(device) for k, v in next(iter(loader)).items()}
    with torch.no_grad():
        plain = model(**first).loss.item()
        masked = model(**first, head_mask=torch.ones(model.config.num_hidden_layers,
                                                     model.config.num_attention_heads, device=device)).loss.item()
    assert abs(plain - masked) < 1e-6, f"head_mask changed the loss ({plain} vs {masked})"

    imp = head_importance(model, loader, device)
    assert torch.isfinite(imp).all() and imp.sum() > 0, "importance is non-finite or all zero (is eager attention on?)"
    norm = normalize_per_layer(imp)

    n_layers, n_heads = imp.shape
    prune = {}
    for name, m in cfg["models"].items():
        frac = m["heads_removed_fraction"]
        if frac > 0:
            k = round(frac * n_layers * n_heads)
            prune[name] = select_heads_to_prune(norm.cpu().numpy(), k, cfg["pruning"]["min_heads_per_layer"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "seed": seed, "importance_split": "train", "n_examples": len(train),
        "importance_raw": imp.cpu().tolist(),
        "importance_layer_l2_normalized": norm.cpu().tolist(),
        "heads_to_prune": {name: {str(l): hs for l, hs in sel.items()} for name, sel in prune.items()},
    }, indent=1))

    print(f"\n[seed {seed}] head importance from {len(train)} training examples -> {out_path}")
    print(f"  zero-importance heads: {int((imp == 0).sum())} of {imp.numel()}")
    print("  layer  mean(norm)  heads kept " + "  ".join(f"{n:>4s}" for n in prune))
    for l in range(n_layers):
        kept = "  ".join(f"{n_heads - len(prune[n].get(l, [])):>4d}" for n in prune)
        print(f"  {l + 1:>5d}  {norm[l].mean().item():>10.3f}  {'':10s} {kept}")


def compare_seeds(seeds):
    """How similar are the head rankings of different seeds? (Spearman correlation of normalised scores)"""
    scores = {}
    for s in seeds:
        p = OUT_DIR / f"M0_seed{s}.json"
        if p.exists():
            scores[s] = np.array(json.loads(p.read_text())["importance_layer_l2_normalized"]).ravel()
    if len(scores) < 2:
        return
    print("\nSpearman correlation of head rankings between seeds (1.0 = identical ranking):")
    keys = sorted(scores)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            print(f"  seed {a} vs seed {b}: {spearmanr(scores[a], scores[b])[0]:.3f}")


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=cfg["project"]["seeds"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    set_fp32_strict()
    device = get_device()
    for seed in args.seeds:
        run_seed(seed, cfg, device, args.force)
    compare_seeds(args.seeds)


if __name__ == "__main__":
    main()
