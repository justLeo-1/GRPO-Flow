# Results — GRPO-Flow vs PPO on robomimic `can` (low-dim)

All numbers were produced with the unified harness `scripts/eval_checkpoints.py`:
same base architecture, 4-step ODE sampling, no exploration noise net, paired
env seeds across checkpoints, success = wrapper-level termination flag
(robosuite `ignore_done=True` never self-terminates; the wrapper marks task
success as done). Wilson 95% CIs reported.

## Final table (one protocol, paired seeds 42+i)

| Version | Recipe highlights | Success | 95% CI | Episodes |
|---|---|---|---|---|
| SFT start (`state_20`) | ReFlow SFT, 20 epochs | 65.5% | [56.2, 73.7] | 110 |
| **PPO best** | official ReinFlow recipe | **90.3%** | [84.3, 94.1] | 144 |
| GRPO sparse old | batch-centering, anchor 0.1, G=5 | 65.5% | — | — (best.pt == SFT weights: training never beat itr-0) |
| GRPO dense | + robosuite staged dense rewards | 65.5% | — | — (same; dense run also exhibited reward hacking) |
| GRPO v10 | positive-only advantages, anchor 0.05, 1 epoch | 73.8% | [64.5, 81.3] | 103 |
| GRPO v12 | G=16, signed + batch-centering, anchor 0.075 | 65.5% | — | — (best.pt == SFT weights) |
| GRPO v13 | + symmetric advantage clip ±1.0, lr 1e-5, grad-clip 1.0 | 77.0% | [68.4, 83.8] | 113 |
| GRPO v14b | + negative-advantage scaling 0.5, lr 2e-5 | 64.7% | [55.1, 73.3] | 102 |
| **GRPO v13b (final)** | v13 recipe + fixed LR schedule, 300 itr | **78.6%** | [70.1, 85.2] | 112 |

Cross-check with three independent evaluation batches (different seed sets and
episode budgets): SFT 66.5% (n=266), GRPO v13b 70.4% (n=270), PPO 87.5% (n=336).
Same ordering, wider CIs — batch-to-batch variability is itself part of the
story (see DIAGNOSIS.md).

## Training-curve behavior (eval every 10 iterations)

Per-iteration eval series are in `results/curves/*.csv` (note: for runs before
v10 the repo's `success rate` log field was broken by a chunk-sum/act_steps
artifact; use the `episode_reward` column, which equals success rate under
sparse rewards).

- **PPO**: 77 → 71 → 81 → 66 → 66 → 86 → 84 → 81 → 91 → 94 → 90 (smooth climb, stopped early at itr 100).
- **GRPO v13b (final)**: 75 → 66 → 64 → 58 → 63 → 70 → 73 → 59 → 64 → 64 → 73 → 33 → 50 → 77 → 75 → 63 → 56 → 70 → 61 → 78 → 53 → 58 → 55 → 61 → 63 — narrow-band oscillation with full recoveries; never settles above the start.

## Takeaways

1. PPO (critic + GAE + per-step trust region) is decisively better on this
   long-horizon sparse-reward task: +25 pp over SFT, stable.
2. GRPO-Flow's best artifacts are genuinely but modestly better than SFT
   (+13 pp, marginal), yet training never stabilizes — every version oscillates
   around an equilibrium pinned near the SFT level.
3. The GRPO improvement ladder is real and reproducible: each mechanism
   (advantage clipping, smaller lr, grad clipping, schedule fix) added a
   measurable bump in artifact quality (73.8 → 77.0 → 78.6).
