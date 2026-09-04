# NodeGuard

**TL;DR:** a graph neural network that flags coordinated mule-account rings
by looking at who an account is connected to, not just its own numbers.
Built in a few days for Razorpay's AI Buildathon. It works (recall 0.333,
precision 0.200 on a held-out test set), but the honest headline is that
the test set only has 3 mule accounts. Treat those numbers as "this
approach is real," not as a production benchmark.

**Live demo: [mule-node-guard.vercel.app](https://mule-node-guard.vercel.app/)**.
A static site in `web/`, built from a JSON export of the real trained
model's predictions. No server, no mocked data.

**For evaluators, fastest way to see the whole feature set:** open the demo,
go to Account Lookup, and try these accounts in order:

- **304** (a caught mule) then click through its neighbors, it's the middle
  of the layering chain.
- **310** (a missed mule) to see a false negative.
- **94** (a false positive the model's own features can explain).
- **171** (a false positive that has no explanation, not in its own
  features, not in its 1-hop or 2-hop neighborhood either).

That last one is the one worth sitting with. It's an honest miss, not a
hidden one.

## The idea, in one paragraph

A mule account often looks unremarkable on its own: normal-ish transaction
count, normal-ish amounts. What gives it away is who it's connected to. A
chain of accounts passing the same money along fast, or a cluster of
ordinary accounts quietly feeding into one collection point. A flat
per-account classifier can't see that, because "my neighbor also looks
weird" isn't a column in a feature table. A graph neural network can,
because it passes information along edges. A node's prediction is built
from its own features and its neighbors'.

## Architecture, briefly

- **Data** (`src/data_gen/`): a synthetic graph, 314 accounts, 300 normal
  and 14 mules split across two real AML typologies: an 8-account
  **layering chain** (money passed fast through a sequence of accounts)
  and a 6-collector **funnel** (many feeders into one collector).
  Synthetic because the buildathon doesn't hand out labeled transaction
  data.
- **Model** (`src/model/gnn.py`): `MuleGCN`, 2 stacked `GCNConv` layers
  (5 to 16 to 2) with dropout. 5 features per account: in/out degree, avg
  in/out amount, total volume. 2 layers specifically because layering
  mules are obvious 1 hop out, but funnel feeders only look wrong once you
  can see their neighbor's in-degree is weird, and that needs the 2nd hop.
- **Training** (`src/model/train.py`): stratified 60/20/20 split (mules are
  only about 4.5% of nodes), class-weighted loss, and checkpointing on
  **validation F1**, not recall. The first version checkpointed on recall
  alone and it locked in the worst possible model (epoch 1, basically
  random weights that flagged almost everything as "mule" and got perfect
  recall by cheating). F1 fixed that. Early stopping on top, patience 20.
- **Eval** (`src/eval/`): `metrics.py` for the headline precision/recall/F1
  and confusion matrix; `error_analysis.py` for the deeper cut, recall
  broken down by fraud pattern, and every false positive traced back to
  either "its own features are genuinely outlier-ish" or "no explanation
  at 1 or 2 hops, probably just the model overfitting on 14 examples."
- **Frontend** (`web/`): plain HTML, CSS and JS, no framework, no server.
  `src/app/export_data.py` runs the real pipeline once and dumps
  predictions and neighborhoods to `web/data.json`. The site is just that
  JSON rendered, deployable anywhere static, including Vercel with zero
  config.

## Drawbacks, said plainly

- **Tiny test set.** 3 mule accounts in test means one flipped prediction
  swings recall by about 33%. The direction (graph structure helps) is
  real; the exact numbers aren't something to defend to three decimal
  places.
- **Synthetic data.** Two hand-built fraud patterns, not real transaction
  behavior. The biggest untested assumption in the whole project is that
  these patterns resemble what real mule rings actually look like.
- **Not built for production.** No latency tuning, no streaming ingestion,
  no incremental retraining. The model is transductive (it needs the whole
  graph at once), so a new account showing up tomorrow isn't something
  this architecture handles gracefully. That's a real rebuild, not a
  tuning pass, if this ever needed to go further.
- **Funnel detection is weak.** 100% layering recall, 0% funnel recall in
  this test split (small sample, so "0%" isn't as damning as it sounds,
  but it's the identified weak point, only 6 funnel examples to learn
  from).
- **Minor scaling leakage.** Feature scaling is fit on the whole graph
  including val/test, not train-only. Leaks scale, not labels. Small, but
  not textbook-correct.



Detection only. Nothing here takes automated action on any account.
