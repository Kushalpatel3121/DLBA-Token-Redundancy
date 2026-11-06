#!/usr/bin/env python3
"""Checks that this machine can run every part of the project.

Run it from the project folder with the venv active:   python scripts/verify_env.py

Each check prints [ OK ], [WARN] or [FAIL]. The script exits with code 1 if anything FAILED.
A summary is written to logs/env_check.json.
"""
import copy
import importlib
import json
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "bert-base-uncased"
results = []


class Warn(Exception):
    """Raise inside a check to report a warning instead of a failure."""


def run(name, fn):
    t0 = time.time()
    try:
        detail = fn()
        status = "OK"
    except Warn as w:
        detail, status = str(w), "WARN"
    except Exception as e:  # noqa: BLE001
        detail, status = f"{type(e).__name__}: {e}", "FAIL"
        if os.environ.get("VERBOSE"):
            detail += "\n" + traceback.format_exc()
    label = {"OK": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[status]
    print(f"{label} {name}: {detail}", flush=True)
    results.append({"check": name, "status": status, "detail": str(detail), "seconds": round(time.time() - t0, 2)})


# ------------------------------------------------------------------ shared state
state = {}


def check_python():
    v = sys.version_info
    assert v >= (3, 10), f"Python {v.major}.{v.minor} is too old (need >= 3.10)"
    return f"{platform.python_version()} on {platform.platform()}"


def check_imports():
    mods = ["torch", "transformers", "datasets", "sklearn", "scipy", "numpy",
            "pandas", "matplotlib", "tqdm", "yaml", "accelerate"]
    versions, missing = [], []
    for m in mods:
        try:
            mod = importlib.import_module(m)
            versions.append(f"{m}={getattr(mod, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001
            missing.append(f"{m} ({type(e).__name__}: {e})")
    if missing:
        raise RuntimeError("cannot import: " + "; ".join(missing))
    import transformers
    major = int(transformers.__version__.split(".")[0])
    if major >= 5:
        raise Warn(f"transformers {transformers.__version__} is 5.x; this project was planned for 4.x "
                   f"(prune_heads / eager attention). If later checks fail, pin transformers<5. " + " ".join(versions))
    return " ".join(versions)


def check_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False. Check `nvidia-smi` and that the "
                           "installed torch wheel matches the driver (see TORCH_INDEX in bootstrap_vm.sh).")
    props = torch.cuda.get_device_properties(0)
    state["device"] = torch.device("cuda:0")
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    t0 = time.time()
    c = a @ b
    torch.cuda.synchronize()
    dt = time.time() - t0
    assert torch.isfinite(c).all().item(), "matmul produced non-finite values"
    return (f"{props.name}, {props.total_memory / 1024**3:.0f} GB, capability {props.major}.{props.minor}, "
            f"torch CUDA {torch.version.cuda}, fp32 4096x4096 matmul {dt * 1000:.1f} ms")


def check_precision_flags():
    import torch
    mm = torch.backends.cuda.matmul.allow_tf32
    cd = torch.backends.cudnn.allow_tf32
    msg = f"matmul.allow_tf32={mm}, cudnn.allow_tf32={cd}"
    if mm or cd:
        raise Warn(msg + " -> project code should set both to False so fp32 really means fp32")
    return msg


def load_model(device):
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, attn_implementation="eager")
    return model.to(device).eval()


def check_model_and_attention():
    import torch
    from transformers import AutoTokenizer
    device = state.get("device", torch.device("cpu"))
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model(device)
    state["tok"], state["model"], state["device"] = tok, model, device
    enc = tok("The company said profits rose.", "Profits at the company went up, it said.",
              return_tensors="pt").to(device)
    state["enc"] = enc
    with torch.no_grad():
        out = model(**enc, output_attentions=True, output_hidden_states=True)
    L = enc["input_ids"].shape[1]
    assert out.attentions is not None and out.attentions[0] is not None, \
        "attention weights not returned: the model is not using the eager attention backend"
    assert len(out.attentions) == 12, f"expected 12 attention tensors, got {len(out.attentions)}"
    assert tuple(out.attentions[0].shape) == (1, 12, L, L), out.attentions[0].shape
    assert len(out.hidden_states) == 13, f"expected 13 hidden states, got {len(out.hidden_states)}"
    row_sums = out.attentions[3].sum(-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), "attention rows do not sum to 1"
    return f"loaded {MODEL_NAME}; 12 attention maps of shape (1,12,{L},{L}); 13 hidden states"


def check_layer_loop():
    """The token-removal experiment runs the encoder layer by layer. Prove that works here."""
    import torch
    model, enc = state["model"], state["enc"]
    bert = model.bert
    with torch.no_grad():
        ref = model(**enc).logits
        h = bert.embeddings(input_ids=enc["input_ids"], token_type_ids=enc["token_type_ids"])
        mask = (1.0 - enc["attention_mask"][:, None, None, :].to(h.dtype)) * torch.finfo(h.dtype).min
        for layer in bert.encoder.layer:
            out = layer(h, attention_mask=mask)
            h = out[0] if isinstance(out, (tuple, list)) else out
        logits = model.classifier(bert.pooler(h))
    diff = (ref - logits).abs().max().item()
    assert diff < 1e-4, (f"manual layer loop differs from the normal forward by {diff:.2e}. "
                         "The layer call signature may differ in this transformers version.")
    return f"manual layer-by-layer forward matches the normal forward (max diff {diff:.1e})"


def check_prune_heads():
    import torch
    model, enc = state["model"], state["enc"]
    pruned = copy.deepcopy(model)
    before = sum(p.numel() for p in pruned.parameters())
    pruned.prune_heads({0: [0, 1], 5: [3]})
    after = sum(p.numel() for p in pruned.parameters())
    assert after < before, "parameter count did not drop after prune_heads"
    with torch.no_grad():
        out = pruned(**enc, output_attentions=True)
    heads = [a.shape[1] for a in out.attentions]
    assert heads[0] == 10 and heads[5] == 11 and all(h == 12 for i, h in enumerate(heads) if i not in (0, 5)), \
        f"unexpected heads per layer after pruning: {heads}"
    del pruned
    return f"prune_heads works: params {before:,} -> {after:,}; heads per layer {heads}"


def check_dataset():
    from datasets import load_dataset
    try:
        ds = load_dataset("glue", "mrpc")
    except Exception as first:  # noqa: BLE001
        try:
            ds = load_dataset("nyu-mll/glue", "mrpc")
        except Exception:
            raise RuntimeError(f"cannot download MRPC ({type(first).__name__}: {first}). Check the VM's "
                               "internet access, or download on your laptop and copy it over.")
    sizes = {k: len(v) for k, v in ds.items()}
    assert sizes.get("train") == 3668 and sizes.get("validation") == 408, f"unexpected split sizes: {sizes}"
    cols = set(ds["train"].column_names)
    assert {"sentence1", "sentence2", "label"} <= cols, f"unexpected columns: {cols}"
    assert set(ds["train"]["label"]) == {0, 1}
    return f"MRPC loaded: {sizes} (train will be split into train + 10% validation; 'validation' here is the official dev set)"


def check_stopwords():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    assert "the" in ENGLISH_STOP_WORDS and "was" in ENGLISH_STOP_WORDS
    return f"scikit-learn stopword list available ({len(ENGLISH_STOP_WORDS)} words)"


def check_folders():
    needed = ["configs", "scripts", "src", "data/splits", "checkpoints/M0", "checkpoints/M0plus",
              "checkpoints/M25", "checkpoints/M50", "cache/hidden_states", "results/labels",
              "results/signals", "results/metrics", "logs", "figures"]
    missing = [d for d in needed if not (ROOT / d).is_dir()]
    assert not missing, f"missing folders: {missing} (run: bash scripts/make_dirs.sh)"
    assert (ROOT / "configs" / "config.yaml").is_file(), "configs/config.yaml is missing"
    probe = ROOT / "results" / ".write_test"
    probe.write_text("ok")
    probe.unlink()
    return "all project folders exist and results/ is writable"


def check_config():
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    assert cfg["project"]["attn_implementation"] == "eager"
    assert cfg["project"]["seeds"] == [42, 123, 2026]
    assert cfg["recovery_finetune"]["extra_epochs"] == 2
    return "configs/config.yaml parses and has the agreed settings"


def check_disk():
    free_gb = shutil.disk_usage(ROOT).free / 1024**3
    if free_gb < 20:
        raise Warn(f"only {free_gb:.0f} GB free; checkpoints plus caches can need ~10-20 GB")
    return f"{free_gb:.0f} GB free"


def main():
    print(f"Project root: {ROOT}\n")
    run("python version", check_python)
    run("package imports", check_imports)
    run("GPU + fp32 compute", check_gpu)
    run("fp32 precision flags", check_precision_flags)
    run("model load + attention weights", check_model_and_attention)
    if "model" in state:
        run("manual layer-by-layer forward", check_layer_loop)
        run("attention-head pruning", check_prune_heads)
    else:
        print("[SKIP] layer loop and pruning checks (model did not load)")
    run("MRPC dataset", check_dataset)
    run("stopword list", check_stopwords)
    run("project folders", check_folders)
    run("config file", check_config)
    run("disk space", check_disk)

    n_fail = sum(r["status"] == "FAIL" for r in results)
    n_warn = sum(r["status"] == "WARN" for r in results)
    print(f"\nSummary: {len(results) - n_fail - n_warn} ok, {n_warn} warnings, {n_fail} failed")
    try:
        (ROOT / "logs").mkdir(exist_ok=True)
        (ROOT / "logs" / "env_check.json").write_text(json.dumps(
            {"python": platform.python_version(), "results": results}, indent=2))
    except OSError as e:
        print(f"(could not write logs/env_check.json: {e})")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
