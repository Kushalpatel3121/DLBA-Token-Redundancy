"""Create the saved train/validation split of MRPC. Run ONCE; the split must never change.

    python src/make_split.py            # refuses to overwrite an existing split
"""
import argparse

import numpy as np

from common import create_split, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite an existing split (avoid!)")
    args = parser.parse_args()

    cfg = load_config()
    payload, labels = create_split(cfg, force=args.force)
    tr, va = np.array(payload["train_indices"]), np.array(payload["val_indices"])
    print(f"Saved split to {cfg['data']['split_file']}")
    print(f"  train: {len(tr)} pairs, {labels[tr].mean():.3f} positive")
    print(f"  val  : {len(va)} pairs, {labels[va].mean():.3f} positive")
    print(f"  total: {payload['n_total']} (no overlap checked)")


if __name__ == "__main__":
    main()
