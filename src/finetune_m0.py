"""Fine-tune BERT-base on MRPC to produce M0, one checkpoint per seed.

    python src/finetune_m0.py --smoke --seeds 42   # ~1 minute pipeline test, writes to checkpoints/_smoke
    python src/finetune_m0.py                      # all seeds from configs/config.yaml
    python src/finetune_m0.py --seeds 123 --force  # redo one seed

Epoch selection uses the validation split only. The official dev set is evaluated once, after
selection, purely to report the number.
"""
import argparse
import gc
import json
import time

import torch
import transformers
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from common import (ROOT, evaluate, get_device, get_logger, get_tokenizer, load_config, load_splits,
                    make_loader, set_fp32_strict, set_seed)

EVAL_BATCH_SIZE = 64
MAX_GRAD_NORM = 1.0  # standard gradient clipping for BERT fine-tuning


def build_optimizer(model, lr, weight_decay):
    # No weight decay on biases and LayerNorm weights (standard BERT recipe).
    no_decay = ("bias", "LayerNorm.weight")
    named = list(model.named_parameters())
    groups = [
        {"params": [p for n, p in named if not any(k in n for k in no_decay)], "weight_decay": weight_decay},
        {"params": [p for n, p in named if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def run_seed(seed, cfg, device, force=False, smoke=False):
    out_dir = ROOT / "checkpoints" / ("_smoke" if smoke else "M0") / f"seed{seed}"
    if (out_dir / "metrics.json").exists() and not force:
        print(f"[seed {seed}] already done ({out_dir}), skipping. Use --force to redo.")
        return
    log = get_logger(f"m0_seed{seed}{'_smoke' if smoke else ''}",
                     ROOT / "logs" / f"m0_seed{seed}{'_smoke' if smoke else ''}.log")

    ft = cfg["finetune_m0"]
    epochs = 1 if smoke else int(ft["epochs"])
    lr, bs = float(ft["learning_rate"]), int(ft["batch_size"])

    set_seed(seed)  # before model creation, so the classifier head init depends on the seed
    tokenizer = get_tokenizer(cfg)
    data = load_splits(cfg, tokenizer)
    if smoke:
        data["train"] = data["train"].select(range(256))
    train_loader = make_loader(data["train"], tokenizer, bs, shuffle=True, seed=seed)
    val_loader = make_loader(data["val"], tokenizer, EVAL_BATCH_SIZE)
    dev_loader = make_loader(data["dev"], tokenizer, EVAL_BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["project"]["model_name"], num_labels=2,
        attn_implementation=cfg["project"]["attn_implementation"]).to(device)
    optimizer = build_optimizer(model, lr, float(ft["weight_decay"]))
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(float(ft["warmup_ratio"]) * total_steps), total_steps)

    log.info(f"seed={seed} smoke={smoke} train={len(data['train'])} val={len(data['val'])} "
             f"dev={len(data['dev'])} epochs={epochs} lr={lr} bs={bs} steps={total_steps}")

    history, best_score, best_epoch, best_state = [], -1.0, 0, None
    t0 = time.time()
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
        val = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": loss_sum / n, **{f"val_{k}": v for k, v in val.items()}})
        log.info(f"epoch {epoch}/{epochs} train_loss={loss_sum / n:.4f} val_loss={val['loss']:.4f} "
                 f"val_acc={val['accuracy']:.4f} val_f1={val['f1']:.4f}")
        if val["score"] > best_score:  # strict '>' keeps the earliest best epoch on ties
            best_score, best_epoch = val["score"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_seconds = time.time() - t0

    model.load_state_dict(best_state)
    dev = evaluate(model, dev_loader, device)  # reporting only, never used for any decision
    log.info(f"best epoch {best_epoch} (val score {best_score:.4f}) | DEV acc={dev['accuracy']:.4f} f1={dev['f1']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    # metrics.json is written last: its presence marks the seed as complete.
    (out_dir / "metrics.json").write_text(json.dumps({
        "seed": seed, "smoke": smoke, "best_epoch": best_epoch, "best_val_score": best_score,
        "history": history, "dev_final": dev, "train_seconds": round(train_seconds, 1),
        "hyperparameters": {"epochs": epochs, "lr": lr, "batch_size": bs, **{k: ft[k] for k in ("warmup_ratio", "weight_decay")}},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
    }, indent=2))
    log.info(f"saved to {out_dir} ({train_seconds / 60:.1f} min of training)")

    del model, optimizer, scheduler, best_state
    gc.collect()
    torch.cuda.empty_cache()


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=cfg["project"]["seeds"])
    parser.add_argument("--force", action="store_true", help="redo seeds that are already finished")
    parser.add_argument("--smoke", action="store_true", help="tiny 1-epoch test run into checkpoints/_smoke")
    args = parser.parse_args()

    set_fp32_strict()
    device = get_device()
    for seed in args.seeds:
        run_seed(seed, cfg, device, force=args.force, smoke=args.smoke)


if __name__ == "__main__":
    main()
