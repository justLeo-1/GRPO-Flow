# GRPO-Flow: Critic-Free RL Fine-Tuning of Flow-Matching Robot Policies

> **Built on [ReinFlow](https://github.com/ReinFlow/ReinFlow) (NeurIPS 2025)** — see `README_ReinFlow.md` for the original project. This repo contains the full ReinFlow codebase plus our GRPO additions, so it runs out of the box.

## What this is

An implementation of **GRPO (Group Relative Policy Optimization) for flow-matching robot policies** — replacing PPO's critic + GAE with group-relative advantage estimation over multiple rollouts that share the same initial state — and a **controlled empirical study** against ReinFlow's PPO baseline on the robomimic `can` (PickPlaceCan, low-dim) task.

The question: *can a critic-free, LLM-style group-relative recipe match PPO for online fine-tuning of flow policies on long-horizon, sparse-reward manipulation?*

**Short answer: not fully.** PPO improves the SFT policy decisively and stably; GRPO-Flow reaches a real but modest and oscillation-prone improvement. The value of this repo is the *why*: a systematic failure-mode diagnosis, two upstream bug fixes, and one environment-reward exploit finding — all documented with reproducible evidence.

## Headline results (unified evaluation protocol)

All checkpoints evaluated under one protocol (same architecture, 4-step ODE sampling, no exploration noise, paired seeds, termination-flag success, ~100–144 episodes per checkpoint, Wilson 95% CI):

| Checkpoint | Success rate | 95% CI |
|---|---|---|
| SFT starting point (behavior cloning / ReFlow) | 65.5% | [56.2, 73.7] |
| **PPO best** (ReinFlow baseline recipe) | **90.3%** | [84.3, 94.1] |
| **GRPO-Flow best (v13b recipe)** | **78.6%** | [70.1, 85.2] |
| GRPO-Flow v13 (same recipe, earlier run) | 77.0% | [68.4, 83.8] |
| GRPO-Flow v10 (positive-only advantages) | 73.8% | [64.5, 81.3] |
| GRPO-Flow v14b (neg-advantage scaling, 2× lr) | 64.7% | [55.1, 73.3] |
| GRPO-Flow v12 / sparse-old / dense | 65.5% | — | their best.pt *is* the SFT checkpoint (training never beat it) |

Full version history, per-iteration curves (CSV), and the evaluation harness: [`docs/RESULTS.md`](docs/RESULTS.md), [`results/curves/`](results/curves/), [`scripts/eval_checkpoints.py`](scripts/eval_checkpoints.py).

## Key findings (the diagnosis)

1. **Outcome-level group advantages cause blanket punishment**: in 300-step sparse-reward episodes, a failed episode's (often correct) prefix gets suppressed along with its mistakes. Symptom: evaluation dips of 20+ points mid-training followed by recovery.
2. **The scalar KL-to-SFT anchor has no workable window**: too strong → frozen policy; too weak → collapse; in between → oscillation around an equilibrium pinned at SFT performance. The anchor cannot tell "good drift" from "bad drift".
3. **Two upstream bugs found and fixed**:
   - `visualize_lr()` dry-runs the *live* LR scheduler for `n_train_itr` steps at startup without resetting it (only `CustomScheduler` was reset) — training silently starts mid-schedule, and with short cosine cycles this triggers an unintended warm restart mid-run. Fixed in `agent/finetune/reinflow/train_ppo_agent.py` and `agent/finetune/grpo/train_grpo_flow_agent.py` via state snapshot/restore.
   - The `reward_shaping` config flag was written to the wrong level of `env_meta` and never reached robosuite — dense rewards silently never activated. Fixed in `env/gym_utils/__init__.py`.
4. **robosuite's staged dense reward is exploitable**: per-step stage payments plus a single terminal +1.0 make *loitering* (grasp/hover without placing) more rewarding than completing the task. Empirically: success rate collapsed (66→26%) while episode return *rose* and episode length approached the time limit.

Details and evidence: [`docs/DIAGNOSIS.md`](docs/DIAGNOSIS.md).

## What we added to ReinFlow

| Path | What |
|---|---|
| `agent/finetune/grpo/train_grpo_flow_agent.py` | `TrainGRPOFlowAgent`: complete-episode rollouts with same-initial-state groups, critic-free updates, plus fixes |
| `agent/finetune/grpo/grpo_buffer.py` | `GRPOFlowBuffer`: group-relative advantages `(R−mean)/(std+eps)`, with `adv_batch_center`, `adv_positive_only`, `adv_clip`, `adv_neg_scale` switches |
| `model/flow/ft_grpo/grpoflow.py` | `GRPOFlow`: PPO-style clipped surrogate over chain log-prob ratios, no value loss |
| `cfg/robomimic/finetune/can/ft_grpo_reflow_mlp.yaml` | final GRPO recipe (v13b) |
| `cfg/robomimic/finetune/can/ft_ppo_reflow_mlp.yaml` | PPO baseline recipe used for the comparison |
| `cfg/robomimic/pretrain/can/pre_reflow_mlp_*.yaml` | SFT recipes with periodic in-simulation evaluation for checkpoint selection |
| `scripts/eval_checkpoints.py` | unified checkpoint evaluation harness (paired seeds, Wilson CI) |
| `scripts/regen_observations.py` | regenerate robomimic low-dim observations with the *local* robosuite (fixes the 1.5.1↔1.4.1 obs-semantics mismatch) |

## Quickstart

Environment setup follows ReinFlow exactly (see `installation/` and `README_ReinFlow.md`).

```bash
# 1. Data: robomimic can PH/MH low_dim hdf5, then rebuild observations with the
#    LOCAL robosuite (see scripts/regen_observations.py docstring for why)
python scripts/regen_observations.py --task can \
    --hdf5 /path/to/can_mh_low_dim_v15.hdf5 --repo-root . --out-dir data/robomimic/can

# 2. SFT (1-Rectified Flow) with periodic in-sim eval for checkpoint selection
python script/run.py --config-dir=cfg/robomimic/pretrain/can --config-name=pre_reflow_mlp_sim

# 3. RL fine-tuning — PPO baseline or our GRPO
python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_ppo_reflow_mlp
python script/run.py --config-dir=cfg/robomimic/finetune/can --config-name=ft_grpo_reflow_mlp

# 4. Head-to-head evaluation under one protocol
python scripts/eval_checkpoints.py --task can \
    --run-config /path/to/a_run/.hydra/config.yaml \
    --ckpt SFT=/path/to/state_20.pt:ema --ckpt PPO=/path/to/best.pt --ckpt GRPO=/path/to/best.pt
```

Hardware note: low-dim runs are CPU-bound (MuJoCo simulation); we used 50–64 parallel envs on 128 CPU cores + one RTX 4090. `n_envs` must be divisible by GRPO `group_size`.

## License & citation

MIT (inherited from ReinFlow; see `LICENSE`). If you use this code, cite ReinFlow (see `CITATION.cff`) and link back to this repo.
