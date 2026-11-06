"""Build M0+, M25 and M50 from each M0 seed: (optionally) prune heads, then fine-tune extra epochs.

    python src/recovery_finetune.py --smoke --seeds 42 --models M25   # ~30 s pipeline test (checkpoints/_smoke)
    python src/recovery_finetune.py                                   # M0plus, M25, M50 for all seeds
    python src/recovery_finetune.py --summary-only                    # just print the results table

M0plus is the control: the same extra training, no pruning. All three models use the identical
recipe (configs/config.yaml -> recovery_finetune): final epoch kept, no checkpoint selection, and
the same data order for a given seed. Heads to prune come from results/head_importance/.
"""
import argparse
import gc
import json
import time

import numpy as np
import torch
import transformers
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from common import (ROOT, evaluate, get_device, get_logger, load_checkpoint, load_config, load_splits,
                    make_loader, set_fp32_strict, set_seed)
from finetune_m0 import EVAL_BATCH_SIZE, MAX_GRAD_NORM, build_optimizer

MODELS = ("M0plus", "M25", "M50")


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def heads_per_layer(model):
    return [layer.attention.self.num_attention_heads for layer in model.bert.encoder.layer]


def run_one(name, seed, cfg, device, force=False, smoke=False):
    out_dir = ROOT / "checkpoints" / ("_smoke" if smoke else "") / name / f"seed{seed}"
    if (out_dir / "metrics.json").exists() and not force:
        print(f"[{name} seed {seed}] already done, skipping. Use --force to redo.")
        return
    base_dir = ROOT / "checkpoints" / "M0" / f"seed{seed}"
    base_meta = json.loads((base_dir / "metrics.json").read_text())

    rec, ft = cfg["recovery_finetune"], cfg["finetune_m0"]
    epochs = 1 if smoke else int(rec["extra_epochs"])
    lr, bs, wd = float(rec["learning_rate"]), int(ft["batch_size"]), float(ft["weight_decay"])
    tag = f"{name}_seed{seed}" + ("_smoke" if smoke else "")
    log = get_logger(tag, ROOT / "logs" / f"{tag}.log")

    ft_seed = seed + 10_000          # same data order for M0plus / M25 / M50 of a given seed
    set_seed(ft_seed)
    tok = AutoTokenizer.from_pretrained(base_dir)
    data = load_splits(cfg, tok)
    if smoke:
        data["train"] = data["train"].select(range(256))
    train_loader = make_loader(data["train"], tok, bs, shuffle=True, seed=ft_seed)
    val_loader = make_loader(data["val"], tok, EVAL_BATCH_SIZE)
    dev_loader = make_loader(data["dev"], tok, EVAL_BATCH_SIZE)

    model = load_checkpoint(base_dir, device)
    params_before = n_params(model)
    frac = cfg["models"][name]["heads_removed_fraction"]
    if frac > 0:
        sel = json.loads((ROOT / "results" / "head_importance" / f"M0_seed{seed}.json").read_text())["heads_to_prune"][name]
        model.prune_heads({int(layer): heads for layer, heads in sel.items()})
    n_total = model.config.num_hidden_layers * model.config.num_attention_heads
    assert n_total - sum(heads_per_layer(model)) == round(frac * n_total), "wrong number of heads removed"
    assert min(heads_per_layer(model)) >= cfg["pruning"]["min_heads_per_layer"]
    params_after = n_params(model)
    pre = evaluate(model, val_loader, device)   # right after pruning, before any recovery training
    log.info(f"{name} seed={seed}: base M0 trained {base_meta['hyperparameters']['epochs']} epochs | "
             f"params {params_before:,} -> {params_after:,} | heads/layer {heads_per_layer(model)}")
    log.info(f"before recovery: val_acc={pre['accuracy']:.4f} val_f1={pre['f1']:.4f}")

    optimizer = build_optimizer(model, lr, wd)   # built AFTER pruning (parameter shapes changed)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)   # linear decay, no warmup
    history, t0 = [], time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.item() * batch["labels"].size(0)
            n += batch["labels"].size(0)
        val = evaluate(model, val_loader, device)   # logged only; the FINAL epoch is always kept
        history.append({"epoch": epoch, "train_loss": loss_sum / n, **{f"val_{k}": v for k, v in val.items()}})
        log.info(f"extra epoch {epoch}/{epochs} train_loss={loss_sum / n:.4f} "
                 f"val_acc={val['accuracy']:.4f} val_f1={val['f1']:.4f}")
    train_seconds = time.time() - t0

    dev = evaluate(model, dev_loader, device)    # reporting only
    log.info(f"FINAL (last epoch) val_acc={val['accuracy']:.4f} | DEV acc={dev['accuracy']:.4f} f1={dev['f1']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    # Guard: the saved checkpoint must reload with the same pruned structure and the same predictions.
    reloaded = load_checkpoint(out_dir, device)
    assert heads_per_layer(reloaded) == heads_per_layer(model), "pruned structure did not survive save/load"
    dev2 = evaluate(reloaded, dev_loader, device)
    assert dev2["accuracy"] == dev["accuracy"] and abs(dev2["loss"] - dev["loss"]) < 1e-4, "reloaded model differs"
    del reloaded

    (out_dir / "metrics.json").write_text(json.dumps({   # written last: marks the run as complete
        "model": name, "seed": seed, "smoke": smoke, "base_m0_epochs": base_meta["hyperparameters"]["epochs"],
        "extra_epochs": epochs, "params_before": params_before, "params_after": params_after,
        "heads_per_layer": heads_per_layer(model), "pre_recovery_val": pre, "history": history,
        "val_final": val, "dev_final": dev, "train_seconds": round(train_seconds, 1),
        "hyperparameters": {"lr": lr, "batch_size": bs, "weight_decay": wd, "scheduler": "linear_no_warmup"},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
    }, indent=2))
    log.info(f"saved to {out_dir} ({train_seconds / 60:.1f} min of training)")

    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()


def summarize():
    """Per-run table plus mean +- sample std over seeds, read from the saved metrics.json files."""
    print(f"\n{'model':7s} {'seed':>5s} {'params':>12s} {'val acc before':>15s} {'val acc final':>14s} "
          f"{'dev acc':>8s} {'dev F1':>7s}")
    agg = {}
    for name in ("M0",) + MODELS:
        for p in sorted((ROOT / "checkpoints" / name).glob("seed*/metrics.json")):
            m = json.loads(p.read_text())
            if name == "M0":
                params, before = "-", "-"
                final = m["history"][m["best_epoch"] - 1]["val_accuracy"]
            else:
                params, before = f"{m['params_after']:,}", f"{m['pre_recovery_val']['accuracy']:.4f}"
                final = m["val_final"]["accuracy"]
            dev = m["dev_final"]
            print(f"{name:7s} {m['seed']:>5d} {params:>12s} {before:>15s} {final:>14.4f} "
                  f"{dev['accuracy']:>8.4f} {dev['f1']:>7.4f}")
            agg.setdefault(name, []).append((dev["accuracy"], dev["f1"]))
    print("\nDev set, mean +- sample std over seeds:")
    for name, vals in agg.items():
        a = np.array(vals)
        sd = a.std(axis=0, ddof=1) if len(a) > 1 else np.zeros(2)
        print(f"  {name:7s} (n={len(a)}) acc {a[:, 0].mean():.4f} +- {sd[0]:.4f}   F1 {a[:, 1].mean():.4f} +- {sd[1]:.4f}")


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=cfg["project"]["seeds"])
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny 1-epoch test into checkpoints/_smoke")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    if not args.summary_only:
        set_fp32_strict()
        device = get_device()
        for seed in args.seeds:
            for name in args.models:
                run_one(name, seed, cfg, device, force=args.force, smoke=args.smoke)
    if not args.smoke:
        summarize()


if __name__ == "__main__":
    main()
