# Mule-Ring GNN — Project Context

## What this is
A graph neural network that detects coordinated mule-account rings from
synthetic transaction data, for the Razorpay AI Buildathon (Track 2: AI Risk Manager).
Accounts = nodes, transactions = edges. Goal: flag structural fraud patterns
a flat per-account feature table wouldn't catch.

## My constraint
I'm a beginner to intermediate to PyTorch Geometric. Explain non-obvious lines, don't just generate
silently. Core model/data-generation logic: I write the first pass myself.
Use Claude Code for: debugging, boilerplate (eval scoring scripts, plotting),
explaining PyG API specifics I haven't hit before.

## Stack
PyTorch, PyTorch Geometric, NetworkX (for synthetic graph gen), scikit-learn (metrics)

## Gotchas to watch for
- PyG graph batching shape mismatches are common — verify tensor shapes at each step
- Synthetic mule-ring pattern must be clean/simple (one clear pattern: rapid
  pass-through chain), not realistic-to-the-point-of-ambiguous
- Held-out test set required — don't eval on training data

## Scope, deliberately
4-day build. If GNN training isn't working correctly by end of day 2,
fallback plan: graph-statistics features (degree, velocity, clustering
coefficient) into a plain logistic regression / random forest instead.
A clean, understood simple version beats a broken ambitious one.

## Deployment 
I want this site to be deployed as a fully fledged MVP that can be transformed into an extension to payment providers like razorpay,visa, billdesk etc. But for the first 4 days I want it to be a competitive MVP that solves the painpoint we are targetting.

## Decision Making 
The decision making should be based on latest industry trends from X.com , reddit.com , job boards , company docs like razorpay while being not limited to them actively search every problem we encounter and what users face. 


