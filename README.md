# NodeGuard

**TL;DR:** a graph neural network that flags coordinated mule-account rings
by looking at *who an account is connected to*, not just its own numbers.
Built in a few days for Razorpay's AI Buildathon. It works — recall 0.333,
precision 0.200 on a held-out test set — but the honest headline is that
the test set only has 3 mule accounts, so treat those numbers as "this
approach is real" evidence, not a production benchmark.

Live demo: a static site in `web/` (deployed on Vercel), built from a JSON
export of the real trained model's predictions — no server, no mocked data.

## The idea, in one paragraph

A mule account often looks unremarkable on its own — normal-ish
transaction count, normal-ish amounts. What gives it away is who it's
connected to: a chain of accounts passing the same money along fast, or a
cluster of ordinary accounts quietly feeding into one collection point. A
flat per-account classifier can't see that, because "my neighbor also
looks weird" isn't a column in a feature table. A graph neural network
can, because it passes information along edges — a node's prediction is
built from its own features *and* its neighbors'.

## Architecture, briefly

- **Data** (`src/data_gen/`): a synthetic graph, 314 accounts — 300 normal,
  14 mules split across two real AML typologies: an 8-account **layering
  chain** (money passed fast through a sequence of accounts) and a
  6-collector **funnel** (many feeders → one collector). Synthetic because
  the buildathon doesn't hand out labeled transaction data.
- **Model** (`src/model/gnn.py`): `MuleGCN`, 2 stacked `GCNConv` layers
  (5 → 16 → 2) with dropout. 5 features per account: in/out degree, avg
  in/out amount, total volume. 2 layers specifically because layering
  mules are obvious 1 hop out, but funnel feeders only look wrong once you
  can see their *neighbor's* in-degree is weird — that needs the 2nd hop.
- **Training** (`src/model/train.py`): stratified 60/20/20 split (mules are
  only ~4.5% of nodes), class-weighted loss, and checkpointing on
  **validation F1**, not recall — the first version checkpointed on recall
  alone and it locked in the *worst* possible model (epoch 1, basically
  random weights that flagged almost everything as "mule" and got perfect
  recall by cheating). F1 fixed that. Early stopping on top, patience 20.
- **Eval** (`src/eval/`): `metrics.py` for the headline precision/recall/F1
  and confusion matrix; `error_analysis.py` for the deeper cut — recall
  broken down by fraud pattern, and every false positive traced back to
  either "its own features are genuinely outlier-ish" or "no explanation
  at 1 or 2 hops, probably just the model overfitting on 14 examples."
- **Frontend** (`web/`): plain HTML/CSS/JS, no framework, no server.
  `src/app/export_data.py` runs the real pipeline once and dumps
  predictions + neighborhoods to `web/data.json`; the site is just that
  JSON rendered — deployable anywhere static, including Vercel with zero
  config.

## Drawbacks — said plainly, not buried

- **Tiny test set.** 3 mule accounts in test means one flipped prediction
  swings recall by ~33%. The direction (graph structure helps) is real;
  the exact numbers aren't something to defend to three decimal places.
- **Synthetic data.** Two hand-built fraud patterns, not real transaction
  behavior. The biggest untested assumption in the whole project is that
  these patterns resemble what real mule rings actually look like.
- **Not built for production.** No latency tuning, no streaming ingestion,
  no incremental retraining — the model is transductive (it needs the
  *whole* graph at once), so a new account showing up tomorrow isn't
  something this architecture handles gracefully. That's a real rebuild,
  not a tuning pass, if this ever needed to go further.
- **Funnel detection is weak.** 100% layering recall, 0% funnel recall in
  this test split (small sample, so "0%" isn't as damning as it sounds,
  but it's the identified weak point — only 6 funnel examples to learn
  from).
- **Minor scaling leakage.** Feature scaling is fit on the whole graph
  (including val/test), not train-only. Leaks scale, not labels — small,
  but not textbook-correct.

## The full report

There's a much longer writeup (`REPORT.md`, gitignored — it's a personal
interview-prep doc, not meant for public consumption) that walks every
file line-by-line, explains every non-obvious design choice, and covers
the debugging story in more depth than makes sense here. This README is
the short version on purpose.

## Running it

```bash
pip install -r requirements.txt

python -m src.data_gen.generate_graph   # sanity-check the synthetic graph
python -m src.model.train               # train + save best checkpoint
python -m src.eval.metrics              # headline precision/recall/F1
python -m src.eval.error_analysis       # per-node / per-pattern deep dive
python -m src.app.export_data           # regenerate web/data.json

cd web && vercel --prod                 # deploy the static demo
```

Detection only — nothing here takes automated action on any account.
