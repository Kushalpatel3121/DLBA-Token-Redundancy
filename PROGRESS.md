# Progress log

Update this file at the end of every session, then run `./scripts/vmctl.sh sync "end of session"`.
On a new VM, read this first: it tells you exactly where to continue.

## Checklist (from the 1-week plan)

### Day 1: setup and M0
- [ ] Environment built on the VM and `verify_env.py` passes
- [ ] `requirements.txt` frozen and committed
- [ ] Saved validation split (`data/splits/mrpc_val_indices.json`)
- [ ] M0 fine-tuned, seeds 42 / 123 / 2026
- [ ] Token-removal function written; no-removal sanity test passes (outputs identical)
- [ ] Head importance computed for each M0 seed

### Day 2: control and pruned models
- [ ] M0+ (2 extra epochs, no pruning), 3 seeds
- [ ] M25 and M50 (prune, then 2 extra epochs), 3 seeds
- [ ] Accuracy, F1, parameters, FLOPs recorded

### Day 3: labels and signals
- [ ] Leave-one-out labels and signals on the validation split (fix signal directions, check epsilon)
- [ ] Same on the dev set, all models

### Day 4 to 7
- [ ] AUROC, Spearman, delta-AUROC vs M0+, redundancy rate, Jaccard, bootstrap CIs
- [ ] Latency measured for all four models in one session
- [ ] Figures 1 to 3
- [ ] Slides drafted
- [ ] Rehearsal and buffer

## Session notes (newest first)

### YYYY-MM-DD: session N
- What was done:
- What is half-finished (and where the files are):
- First thing to do next time:
- Problems / things to remember:
