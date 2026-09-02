# NodeGuard — Mule-Ring Detection with a GNN

Built for Razorpay's AI Buildathon, Track 2 (AI Risk Manager). It flags
coordinated mule-account rings in a transaction network using a graph
neural network, instead of scoring each account in isolation.

## Why a graph, not a flat feature table

A single mule account often doesn't look that suspicious on its own —
normal-ish transaction count, normal-ish amounts. What gives it away is who
it's connected to: a chain of accounts passing the same money along fast,
or a cluster of ordinary accounts all quietly feeding into one collection
point. A flat per-account model can't see that, because "my neighbor also
looks weird" isn't a column in a feature table. A GNN can, because it
literally passes information along edges — a node's prediction is built
from its own features *and* its neighbors'.

This actually showed up in the error analysis (below): a couple of the
model's false positives had nothing wrong in their own numbers and no mule
anywhere in their 1-hop or 2-hop neighborhood either — meaning the model's
mistakes aren't reducible to "the numbers looked mule-ish," which is the
only failure mode a flat classifier would even be capable of.

## Data

Synthetic, generated in `src/data_gen/generate_graph.py` — 314 accounts,
300 normal and 14 mules, split across two real AML (anti-money-laundering)
typologies:

- **Layering chain** (8 accounts): a straight pass-through, account A →
  B → C → ... — money moving through several accounts fast to obscure
  where it came from.
- **Funnel** (6 collector accounts, fed by 15 distinct normal accounts):
  many ordinary-looking accounts each sending a few payments into one
  collector. Any single feeder still looks mostly normal; it's the
  collector's abnormal in-degree that's the tell.

It's synthetic because Razorpay's buildathon doesn't hand out a labeled
transaction dataset — you're expected to build your own. The upside: clean
ground-truth labels with zero annotation noise, which made it possible to
actually trace *why* the model got specific predictions wrong later on.

## Model

`MuleGCN` (`src/model/gnn.py`) — 2 stacked `GCNConv` layers, 5 → 16 → 2,
with dropout in between. 5 node features:
`in_degree, out_degree, avg_in_amount, avg_out_amount, total_volume`.

**Why 2 layers:** each layer lets a node "see" 1 hop further. Layering
mules are locally obvious — 1 hop is already enough, their neighbors are
also obviously weird. Funnel feeders aren't obvious on their own; you only
catch them by noticing their *neighbor* (the collector) has a weirdly high
in-degree, which needs a 2nd hop. 2 is the minimum depth that reaches both
patterns — a 3rd layer was considered and skipped, since with only 14
positive examples, more depth mostly just risks oversmoothing and overfitting.

## What broke, and how it got fixed

The real debugging story here isn't in the model architecture, it's in how
"best model" got defined during training.

First pass at early stopping checkpointed on **validation recall alone**.
Ran it, and the "best" checkpoint it picked was **epoch 1** — the model
right after random init. Turns out an untrained model that flags almost
every node as "mule" gets near-perfect recall for free (you can't miss a
mule if you flag everyone), even though its precision was garbage (0.067 —
93% of its "mule" calls were wrong). Recall alone doesn't punish that.

Fix: checkpoint on **F1** (harmonic mean of precision and recall) instead.
F1 collapses toward zero if either precision or recall is near zero, so
"flag everything" can't fake a good score anymore. Re-ran it, and the real
best checkpoint landed at epoch 5 (F1 0.800, precision 1.000, recall
0.667) — an actual selective model, not a random one that got lucky on one
metric.

Kept early stopping around this (patience of 20 epochs with no F1
improvement), since the training curve also showed real overfitting —
val recall visibly decayed over a full 150-epoch run as training loss kept
dropping, i.e. the model memorizing its 8 training-set mules instead of
generalizing.

## Results

From `src/eval/metrics.py`, on the held-out test set:

| Metric | Value |
|---|---|
| Precision | 0.200 |
| Recall | 0.333 |
| F1 | 0.250 |

Confusion matrix (raw counts — with only 3 mules in test, percentages
alone would be misleading):

|  | predicted mule | predicted normal |
|---|---|---|
| **actual mule** | 1 | 2 |
| **actual normal** | 4 | 56 |

4 normal accounts would get flagged for unnecessary manual review.

**Caveat, stated plainly:** 3 mules in the test set is not enough to treat
these numbers as statistically solid — read them as directional evidence
the approach works, not as a precise performance benchmark. One flipped
prediction moves recall by ~0.33.

## Error analysis

From `src/eval/error_analysis.py`, broken down by pattern:

- **Layering: 1/1 caught (100%)**
- **Funnel: 0/2 caught (0%)**

Matches the architecture reasoning above — funnel is the harder,
2-hop-dependent pattern, and it's the one that underperformed. With only 2
funnel mules in this split, though, "0%" is one lucky draw away from
"50%" — a real but noisy signal, not proof the model can't learn funnel
patterns at all.

The 4 false positives split into two honest categories:

- **2 explainable** (nodes 94, 130) — their raw features genuinely lean
  toward the mule profile (e.g. node 94 has in-degree 0, way outside the
  normal average of ~5.3). Reasonable mistakes, not unexplained ones.
- **2 unexplained** (nodes 171, 219) — normal-looking features, and a full
  1-hop *and* 2-hop neighbor trace (the entire reach of this 2-layer model)
  found no mule connection at either distance. Best explanation: the model
  overgeneralizing from a small training set (14 mules total), not a
  specific structural cause. Not forcing a story the data doesn't support.

## Limitations

- **Small dataset.** 14 mules total is enough to show the approach works,
  not enough to make the metrics above statistically reliable.
- **Not production-tuned.** This wasn't built or benchmarked for inference
  latency, streaming ingestion, or production request volume — it's a
  from-scratch demonstration of the detection approach, evaluated offline.
- **One fixed seed.** Everything runs on seed 42 for reproducibility, which
  also means there's exactly one draw of the graph and one training run
  behind these numbers — no variance across seeds has been measured.
- **Minor scaling leakage.** `StandardScaler` in `to_pyg.py` is fit on all
  314 nodes including val/test, not train-only. Leaks feature scale, not
  labels — minor, but not textbook-correct.
- **No time-based split.** Transactions have timestamps, but the split is
  random-stratified over nodes, not time-based. A real deployment
  predicting on future transactions would need to validate that way instead.

## What I'd do with more time

- More funnel-pattern training examples — the identified weak point, and
  it's a data problem more than a model problem.
- Build the 2-hop error trace into the eval pipeline as a reusable check
  instead of hand-tracing it per false positive.
- Real (or realistically distributed) transaction data to validate that
  these two hand-built typologies actually match real mule-ring structure
  — the biggest untested assumption in the whole project.
- Fit the feature scaler on train only, and add a time-based split
  alongside the current stratified one.

## Running it

```bash
pip install -r requirements.txt

python -m src.data_gen.generate_graph   # sanity-check the synthetic graph
python -m src.model.train               # train + save best checkpoint
python -m src.eval.metrics              # headline precision/recall/F1
python -m src.eval.error_analysis       # per-node / per-pattern deep dive
```
