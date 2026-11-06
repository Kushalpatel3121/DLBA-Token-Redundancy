"""Mid-network token removal for BertForSequenceClassification (single, unpadded examples).

"Remove token i after layer l" means: run layers 1..l normally, delete position i from the hidden
states, run layers l+1..12 on the shorter sequence, then pooler + classifier. Position and segment
information is already baked into the hidden states, so nothing else needs adjusting.

[CLS] (position 0) and [SEP] are never candidates. Because [CLS] is never removed, the pooler
(which reads position 0) always works.
"""
import torch


def example_to_batch(ex, device):
    """One tokenized dataset row (lists) -> batch of size 1, no padding."""
    return {k: torch.tensor([ex[k]], device=device) for k in ("input_ids", "token_type_ids", "attention_mask")}


def candidate_positions(input_ids, tokenizer):
    """Positions that may be removed: everything except [CLS] and [SEP]."""
    ids = input_ids[0]
    special = (ids == tokenizer.cls_token_id) | (ids == tokenizer.sep_token_id)
    return (~special).nonzero(as_tuple=True)[0]


def _call(layer, h, mask=None):
    out = layer(h, attention_mask=mask)
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def hidden_after(model, enc, layers):
    """{l: hidden states after layer l} for each l in `layers` (1-based), computed in one pass."""
    assert not model.training, "call model.eval() first"
    bert = model.bert
    h = bert.embeddings(input_ids=enc["input_ids"], token_type_ids=enc["token_type_ids"])
    out = {}
    for idx, layer in enumerate(bert.encoder.layer[:max(layers)], start=1):
        h = _call(layer, h)
        if idx in layers:
            out[idx] = h
    return out


@torch.no_grad()
def finish(model, h, start, mask=None):
    """Run layers start+1..12 on h, then pooler and classifier. Returns logits (batch, 2)."""
    bert = model.bert
    for layer in bert.encoder.layer[start:]:
        h = _call(layer, h, mask)
    return model.classifier(bert.pooler(h))


@torch.no_grad()
def forward_removing(model, h, start, remove):
    """Logits when the positions in `remove` (possibly empty) are deleted from h after layer `start`."""
    keep = torch.ones(h.shape[1], dtype=torch.bool, device=h.device)
    if len(remove) > 0:
        keep[torch.as_tensor(remove, device=h.device)] = False
    return finish(model, h[:, keep], start)


@torch.no_grad()
def leave_one_out_logits(model, h, start, positions, chunk=256):
    """Logits for each single-token removal, batched. Row k = position positions[k] removed.

    Every row has the same length (L-1), so no padding is needed.
    """
    L = h.shape[1]
    arange = torch.arange(L, device=h.device)
    outs = []
    for s in range(0, len(positions), chunk):
        pos = positions[s:s + chunk]
        keep = arange[None, :] != pos[:, None]                          # (n, L), one False per row
        idx = arange[None, :].expand(len(pos), L)[keep].view(len(pos), L - 1)
        outs.append(finish(model, h[0][idx], start))                     # h[0][idx]: (n, L-1, hidden)
    return torch.cat(outs)


@torch.no_grad()
def leave_one_out(model, enc, tokenizer, layers=(4, 6, 8)):
    """Per-token redundancy scores for one example.

    delta = |p(yhat | full input) - p(yhat | token removed after layer l)|, where yhat is the class
    predicted on the full input. Small delta = the token is (nearly) redundant at that layer.
    """
    model.eval()
    pos = candidate_positions(enc["input_ids"], tokenizer)
    p_full = model(**enc).logits[0].softmax(-1)
    yhat = int(p_full.argmax())
    states = hidden_after(model, enc, layers)
    result = {}
    for l in layers:
        p_removed = leave_one_out_logits(model, states[l], l, pos).softmax(-1)[:, yhat]
        result[l] = {"positions": pos.cpu(), "delta": (p_full[yhat] - p_removed).abs().cpu()}
    return {"yhat": yhat, "p_full": float(p_full[yhat]), "layers": result}
