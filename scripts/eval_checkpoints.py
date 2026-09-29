"""Unified head-to-head checkpoint evaluation for robomimic low-dim tasks.

Evaluates any number of checkpoints under ONE protocol so numbers are
comparable: same base architecture, same sampler (few-step ODE, no exploration
noise net), same env seeds (paired across checkpoints), success read from the
wrapper-level termination flag (robosuite `ignore_done=True` never terminates
on its own; the wrapper marks success as done).

Usage:
    python scripts/eval_checkpoints.py --task can \
        --run-config /path/to/any_run/.hydra/config.yaml \
        --ckpt SFT=/path/to/state_20.pt:ema \
        --ckpt PPO=/path/to/best.pt:policy \
        --ckpt GRPO=/path/to/best.pt:policy \
        --n-envs 50 --n-steps 152 --seed 42

Checkpoint key: finetune checkpoints store the pure flow policy under 'policy';
pretrain (SFT) checkpoints store it under 'ema'. If the ':key' suffix is
omitted, the script auto-detects ('policy' first, then 'ema').
"""
import argparse
import os
import sys

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env.gym_utils import make_async

TASK_WRAPPER_KEYS = {
    "can": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"],
    "transport": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
                  "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos", "object"],
}
TASK_DIMS = {"can": (23, 7), "transport": (59, 14)}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASK_WRAPPER_KEYS), default="can")
    p.add_argument("--run-config", required=True,
                   help="resolved .hydra/config.yaml from any training run of this task "
                        "(used only to instantiate the policy architecture)")
    p.add_argument("--ckpt", action="append", required=True,
                   help="name=/path/to.ckpt[:key] — repeatable; key defaults to auto-detect")
    p.add_argument("--normalization", default=None, help="path to normalization.npz")
    p.add_argument("--n-envs", type=int, default=50)
    p.add_argument("--n-steps", type=int, default=152,
                   help="policy calls per env; more steps = more episodes per env")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    obs_dim, act_dim = TASK_DIMS[args.task]
    act_steps = 4
    max_eps = 300
    norm_path = args.normalization or os.path.join("data/robomimic", args.task, "normalization.npz")

    venv = make_async(
        args.task, env_type=None, num_envs=args.n_envs, asynchronous=True,
        max_episode_steps=max_eps,
        wrappers=OmegaConf.create({
            "robomimic_lowdim": {"normalization_path": norm_path,
                                 "low_dim_keys": TASK_WRAPPER_KEYS[args.task]},
            "multi_step": {"n_obs_steps": 1, "n_action_steps": act_steps,
                           "max_episode_steps": max_eps, "reset_within_step": True},
        }),
        robomimic_env_cfg_path=f"cfg/robomimic/env_meta/{args.task}.json",
        shape_meta=None, use_image_obs=False, render=False, render_offscreen=False,
        obs_dim=obs_dim, action_dim=act_dim,
    )

    cfg = OmegaConf.load(args.run_config)

    for spec in args.ckpt:
        name, _, rest = spec.partition("=")
        path, _, key = rest.partition(":")
        model = hydra.utils.instantiate(cfg.model)
        ckpt = torch.load(path, map_location="cuda:0" if torch.cuda.is_available() else "cpu",
                          weights_only=True)
        if not key:
            key = "policy" if "policy" in ckpt else ("ema" if "ema" in ckpt else "model")
        missing, unexpected = model.load_state_dict(ckpt[key], strict=False)
        print(f"[{name}] key='{key}' missing={len(missing)} unexpected={len(unexpected)}",
              flush=True)
        model.eval()

        venv.seed([args.seed + i for i in range(args.n_envs)])
        obs = venv.reset_arg(options_list=[{} for _ in range(args.n_envs)])
        if isinstance(obs, list):
            obs = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
        firsts = np.zeros((args.n_steps + 1, args.n_envs))
        firsts[0] = 1
        terms = np.zeros((args.n_steps, args.n_envs), dtype=bool)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        for step in range(args.n_steps):
            with torch.no_grad():
                cond = {"state": torch.from_numpy(obs["state"]).float().to(device)}
                a = model.sample(cond, inference_steps=4, record_intermediate=False,
                                 clip_intermediate_actions=True).trajectories.cpu().numpy()
            obs, rew, term, trunc, info = venv.step(a[:, :act_steps])
            terms[step] = term
            firsts[step + 1] = term | trunc

        succ, lens = [], []
        for e in range(args.n_envs):
            idx = np.where(firsts[:, e] == 1)[0]
            for i in range(len(idx) - 1):
                s, en = idx[i], idx[i + 1]
                if en - s > 1:
                    succ.append(bool(terms[s:en, e].any()))
                    lens.append((en - s) * act_steps)
        k, n = int(np.sum(succ)), len(succ)
        lo, hi = wilson(k, n)
        print(f"RESULT {name}: success={100*k/n:.1f}%  (Wilson95 [{100*lo:.1f}, {100*hi:.1f}])  "
              f"ep_len={np.mean(lens):.0f}  n_episodes={n}", flush=True)

    print("ALL_DONE")


if __name__ == "__main__":
    main()
