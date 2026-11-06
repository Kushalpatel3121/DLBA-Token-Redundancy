"""Shared helpers used by every script: config, seeding, strict fp32, MRPC splits, loaders, evaluation."""
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def set_fp32_strict():
    """fp32 must mean true fp32: switch off TF32 shortcuts (epsilon = 0.01 is noise-sensitive)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Not fully bit-exact across runs, but removes the main sources of nondeterminism.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU visible. Refusing to run on CPU; check nvidia-smi and the venv.")
    return torch.device("cuda:0")


def get_logger(name, logfile=None):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    handlers = [logging.StreamHandler()]
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile))
    for h in handlers:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def load_checkpoint(path, device, attn_implementation="eager"):
    """Load a saved model (also works for head-pruned checkpoints) in eval mode on the device."""
    model = AutoModelForSequenceClassification.from_pretrained(path, attn_implementation=attn_implementation)
    return model.to(device).eval()


# ---------------------------------------------------------------- data
def load_mrpc_raw():
    from datasets import load_dataset
    try:
        return load_dataset("glue", "mrpc")
    except Exception:  # noqa: BLE001  (newer hub layout)
        return load_dataset("nyu-mll/glue", "mrpc")


def split_path(cfg):
    return ROOT / cfg["data"]["split_file"]


def create_split(cfg, force=False):
    """Make the stratified train/validation split of the MRPC train set ONCE and save the indices."""
    path = split_path(cfg)
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. The split must never change; use --force only on purpose.")
    train = load_mrpc_raw()["train"]
    labels = np.array(list(train["label"]))
    train_idx, val_idx = train_test_split(
        np.arange(len(train)),
        test_size=cfg["data"]["validation_fraction"],
        random_state=cfg["data"]["split_seed"],
        stratify=labels,
    )
    train_idx, val_idx = sorted(train_idx.tolist()), sorted(val_idx.tolist())
    assert not set(train_idx) & set(val_idx) and len(train_idx) + len(val_idx) == len(train)
    payload = {
        "n_total": len(train),
        "validation_fraction": cfg["data"]["validation_fraction"],
        "split_seed": cfg["data"]["split_seed"],
        "train_indices": train_idx,
        "val_indices": val_idx,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload, labels


def get_tokenizer(cfg):
    return AutoTokenizer.from_pretrained(cfg["project"]["model_name"])


def load_splits(cfg, tokenizer):
    """Returns tokenized datasets: train (90% of MRPC train), val (10%), dev (official GLUE validation)."""
    raw = load_mrpc_raw()
    sp = json.loads(split_path(cfg).read_text())
    assert len(raw["train"]) == sp["n_total"], "MRPC train size changed since the split was saved"
    max_len = cfg["project"]["max_length"]
    keep = {"input_ids", "token_type_ids", "attention_mask", "labels"}

    def tok(batch):
        return tokenizer(batch["sentence1"], batch["sentence2"], truncation=True, max_length=max_len)

    def prep(ds):
        ds = ds.map(tok, batched=True).rename_column("label", "labels")
        return ds.remove_columns([c for c in ds.column_names if c not in keep])

    return {
        "train": prep(raw["train"].select(sp["train_indices"])),
        "val": prep(raw["train"].select(sp["val_indices"])),
        "dev": prep(raw["validation"]),
    }


def make_loader(ds, tokenizer, batch_size, shuffle=False, seed=0):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=gen if shuffle else None,
        collate_fn=DataCollatorWithPadding(tokenizer),  # dynamic padding to the longest in the batch
    )


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, loader, device):
    """Loss, accuracy, F1 (positive class = paraphrase) and their mean ('score') on a loader."""
    model.eval()
    total_loss, n = 0.0, 0
    preds, labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        bs = batch["labels"].size(0)
        total_loss += out.loss.item() * bs
        n += bs
        preds.append(out.logits.argmax(-1).cpu())
        labels.append(batch["labels"].cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    acc, f1 = accuracy_score(labels, preds), f1_score(labels, preds)
    return {"loss": total_loss / n, "accuracy": float(acc), "f1": float(f1), "score": float((acc + f1) / 2)}
