"""Sanity tests for the token-removal code, run on a saved M0 checkpoint.

    python src/test_token_removal.py --seed 42 --n 20

Three independent checks (all must be ~0, below 1e-4 on the logits):
  identity         removing NO tokens reproduces the normal forward pass
  batch_vs_single  the batched leave-one-out equals removing one token at a time
  delete_vs_mask   deleting token i after layer l equals keeping it but masking it as an attention
                   key in layers l+1..12 (mathematically identical for all other tokens)

Then it prints a first look at redundancy rates and the speed of leave-one-out.
"""
import argparse
import sys
import time

import torch
from transformers import AutoTokenizer

from common import ROOT, get_device, load_checkpoint, load_config, load_splits, set_fp32_strict
from token_removal import (candidate_positions, example_to_batch, finish, forward_removing, hidden_after,
                           leave_one_out, leave_one_out_logits)

TOL = 1e-4


def maxdiff(a, b):
    return (a - b).abs().max().item()


def key_mask(h, i):
    """Additive attention mask that hides position i as a key, shape (1, 1, 1, L)."""
    m = torch.zeros(1, 1, 1, h.shape[1], device=h.device, dtype=h.dtype)
    m[..., i] = torch.finfo(h.dtype).min
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=20, help="number of examples to test")
    parser.add_argument("--split", default="val", choices=["val", "dev"])
    args = parser.parse_args()

    cfg = load_config()
    set_fp32_strict()
    device = get_device()
    ckpt = ROOT / "checkpoints" / "M0" / f"seed{args.seed}"
    model = load_checkpoint(ckpt, device)
    tok = AutoTokenizer.from_pretrained(ckpt)
    ds = load_splits(cfg, tok)[args.split]
    layers = tuple(cfg["redundancy"]["drop_layers"])
    eps_list = [cfg["redundancy"]["epsilon_main"]] + list(cfg["redundancy"]["epsilon_sensitivity"])

    worst = {"identity": 0.0, "batch_vs_single": 0.0, "delete_vs_mask": 0.0}
    deltas = {l: [] for l in layers}
    confidences, loo_seconds, n_done = [], 0.0, min(args.n, len(ds))

    for i in range(n_done):
        enc = example_to_batch(ds[i], device)
        with torch.no_grad():
            ref = model(**enc).logits
            states = hidden_after(model, enc, layers)
            pos = candidate_positions(enc["input_ids"], tok)
            for l in layers:
                h = states[l]
                worst["identity"] = max(worst["identity"], maxdiff(forward_removing(model, h, l, []), ref))
                batched = leave_one_out_logits(model, h, l, pos)
                single = torch.cat([forward_removing(model, h, l, [int(p)]) for p in pos])
                masked = torch.cat([finish(model, h, l, mask=key_mask(h, int(p))) for p in pos])
                worst["batch_vs_single"] = max(worst["batch_vs_single"], maxdiff(batched, single))
                worst["delete_vs_mask"] = max(worst["delete_vs_mask"], maxdiff(single, masked))

        torch.cuda.synchronize()
        t0 = time.time()
        res = leave_one_out(model, enc, tok, layers)
        torch.cuda.synchronize()
        loo_seconds += time.time() - t0

        confidences.append(res["p_full"])
        for l in layers:
            d = res["layers"][l]["delta"]
            assert torch.isfinite(d).all() and (d >= 0).all() and (d <= 1).all(), "delta outside [0, 1]"
            deltas[l].append(d)

    print(f"\nCheckpoint: {ckpt}   split: {args.split}   examples: {n_done}   layers: {list(layers)}")
    print("\nCorrectness (max abs logit difference, must be < %g):" % TOL)
    ok = True
    for name, v in worst.items():
        status = "PASS" if v < TOL else "FAIL"
        ok &= v < TOL
        print(f"  [{status}] {name:16s} {v:.2e}")

    print(f"\nFirst look (informational): mean confidence in the predicted class = {sum(confidences) / n_done:.3f}")
    print("Fraction of candidate tokens redundant (delta < epsilon):")
    print("  layer " + "".join(f"eps={e:<8g}" for e in eps_list) + "median delta")
    for l in layers:
        d = torch.cat(deltas[l])
        print(f"  {l:<6d}" + "".join(f"{(d < e).float().mean().item():<12.3f}" for e in eps_list) + f"{d.median().item():.4f}")

    per_ex = loo_seconds / n_done
    print(f"\nSpeed: {per_ex:.3f} s per example (all {len(layers)} layers). "
          f"Estimate for 408 dev examples: {per_ex * 408 / 60:.1f} min per model.")
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
