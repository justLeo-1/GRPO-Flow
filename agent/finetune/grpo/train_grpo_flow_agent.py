# MIT License

# Copyright (c) 2025 ReinFlow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
GRPO-Flow: critic-free Group Relative Policy Optimization fine-tuning of
flow-matching robot policies.

GRPO mapping to this domain:
    prompt                <- initial environment state s_0
    group of completions  <- G complete episodes rolled out from the SAME s_0:
        the n_envs workers are partitioned into n_groups = n_envs // group_size
        groups; at each training iteration every worker of a group is reset with
        the SAME seed (-> identical initial state), different groups get
        different seeds, all drawn from a base rng seeded by cfg.seed + itr.
    outcome reward        <- discounted episode return R_i = sum_t gamma^t r_t
    advantage             <- group-relative, A_i = (R_i - mean_g) / (std_g + eps),
        broadcast to every step of episode i (outcome supervision).

Why no critic: the baseline of the ratio objective is the group mean return, so
there is no value network, no GAE, no value loss, no critic optimizer, and no
fixed-length truncated rollout segments -- rollouts are complete episodes
stored in a variable-length episode buffer (GRPOFlowBuffer).

Per iteration:
    1. grouped identical-state reset (train) or plain reset (eval)
    2. roll complete episodes with the current policy (exploration noise on,
       same sampling path as the PPO flow baseline), masking envs once done
    3. R_i per trajectory, group-relative A_i broadcast per step
    4. for update_epochs: clipped surrogate over chain-logprob ratios
       (PPOFlow.get_logprobs recomputation) + entropy bonus, actor-only
       optimizer step, early stop on approx_kl > target_kl
    5. exploration-noise scheduling, lr scheduling, logging, checkpointing

Reused from the PPO baseline because it is algorithm-shared (not PPO-specific):
chain sampling/logprob machinery, NoisyFlowMLP exploration noise and its
per-iteration scheduling, LR-scheduler types, eval/checkpoint/wandb patterns.
"""
import os
import pickle
import logging
log = logging.getLogger(__name__)
from typing import Optional
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt

from agent.finetune.reinflow.train_agent import TrainAgent
from agent.finetune.grpo.grpo_buffer import GRPOFlowBuffer
from model.flow.ft_grpo.grpoflow import GRPOFlow
from util.scheduler import CosineAnnealingWarmupRestarts, WarmupReduceLROnPlateau
from util.scheduler_simple import get_scheduler, CustomScheduler
from util.timer import Timer
from util.logging_custom import create_bordered_text


class TrainGRPOFlowAgent(TrainAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model: GRPOFlow

        # GRPO grouping: envs [g*group_size, (g+1)*group_size) form group g
        self.group_size: int = cfg.train.group_size
        assert (
            self.n_envs % self.group_size == 0
        ), f"n_envs={self.n_envs} must be divisible by group_size={self.group_size}"
        self.n_groups = self.n_envs // self.group_size
        self.adv_eps: float = cfg.train.get("adv_eps", 1e-6)
        self.adv_use_std_norm: bool = cfg.train.get("adv_use_std_norm", True)

        self.gamma: float = cfg.train.gamma
        self.target_kl: Optional[float] = cfg.train.target_kl
        self.update_epochs: int = cfg.train.update_epochs
        self.ent_coef: float = cfg.train.get("ent_coef", 0.01)

        # Optional regularization to the frozen reference policy actor_old.
        # Default OFF, mirroring the PPO baseline yaml (keeps comparison clean).
        self.use_ref_reg: bool = cfg.train.get("use_ref_reg", False)
        self.ref_reg_coef: float = cfg.train.get("ref_reg_coef", 0.0)

        # Actor-only optimizer (there is no critic anywhere in GRPO)
        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor_ft.parameters(),
            lr=cfg.train.actor_lr,
            weight_decay=cfg.train.actor_weight_decay,
        )
        self.actor_lr_type = cfg.train.actor_lr_scheduler.get("type", "cosine")
        if self.actor_lr_type == "cosine":
            self.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.actor_optimizer,
                first_cycle_steps=cfg.train.actor_lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.train.actor_lr,
                min_lr=cfg.train.actor_lr_scheduler.min_lr,
                warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
                gamma=1.0,
            )
        elif self.actor_lr_type == "plateau":
            self.actor_lr_scheduler = WarmupReduceLROnPlateau(
                self.actor_optimizer,
                warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
                target_lr=cfg.train.actor_lr,
                mode='max',
                min_lr=cfg.train.actor_lr_scheduler.min_lr,
                factor=0.6,
                patience=4,
                threshold=20,
                verbose=True,
            )
        elif self.actor_lr_type == 'constant_warmup':
            self.actor_lr_scheduler = CustomScheduler(
                self.actor_optimizer,
                'constant_warmup',
                min=cfg.train.actor_lr_scheduler.min_lr,
                warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
                max=cfg.train.actor_lr,
            )
        elif self.actor_lr_type == 'cosine_custom':
            self.actor_lr_scheduler = CustomScheduler(
                self.actor_optimizer,
                schedule_type='cosine',
                max=cfg.train.actor_lr,
                hold_steps=cfg.train.actor_lr_scheduler.hold_steps,
                anneal_steps=cfg.train.actor_lr_scheduler.anneal_steps,
                min=cfg.train.actor_lr_scheduler.min_lr,
            )
        else:
            raise ValueError(f"Invalid actor_lr_type: {self.actor_lr_type}")
        self.visualize_lr(cfg)

        self.lr_schedule = cfg.train.lr_schedule
        if self.lr_schedule not in ["fixed", "adaptive_kl"]:
            raise ValueError("lr_schedule should be 'fixed' or 'adaptive_kl'")
        self.actor_lr = cfg.train.actor_lr

        # Flow sampling / logprob flags, identical to the PPO flow sampling path
        self.skip_initial_eval = cfg.get('skip_initial_eval', False)
        self.inference_steps = self.model.inference_steps
        self.ft_denoising_steps = self.model.ft_denoising_steps
        self.normalize_act_space_dim = True   # normalize entropy and logprob over horizon steps and action dim
        self.normalize_denoising_horizon = True   # normalize denoising horizon in chain logprob
        self.clip_intermediate_actions = cfg.train.get("clip_intermediate_actions", True)
        self.account_for_initial_stochasticity = cfg.train.get('account_for_initial_stochasticity', True)

        # Exploration-noise scheduling (ported from the PPO flow agent; it
        # drives NoisyFlowMLP's noise net and is orthogonal to the RL algorithm)
        if self.model.noise_scheduler_type == 'const_schedule_itr':
            self.explore_noise_scheduler = get_scheduler(schedule_type='cosine_warmup',
                                                            min=0.016,
                                                            warmup_steps=self.n_train_itr * 0.01,
                                                            max=0.08,
                                                            hold_steps=self.n_train_itr * 0.29,
                                                            anneal_steps=self.n_train_itr * 0.7)
            explore_noises = [self.explore_noise_scheduler(t) for t in np.arange(self.n_train_itr)]
            plt.figure()
            plt.plot(np.arange(self.n_train_itr), explore_noises)
            name = os.path.join(self.logdir, 'explore_noise') + '.png'
            plt.savefig(name)
            plt.close()
            log.info("Exploration noise saved to %s" % name)
        elif self.model.noise_scheduler_type == 'learn_decay':
            max_std = cfg.model.max_logprob_denoising_std
            min_std = cfg.model.min_logprob_denoising_std
            self.max_noise_decay_ratio = cfg.train.get('max_noise_decay_ratio', 0.7)
            max_std_decayed = min_std * (1 - self.max_noise_decay_ratio) + max_std * self.max_noise_decay_ratio
            self.max_noise_hold_ratio = cfg.train.get('max_noise_hold_ratio', 0.35)
            self.explore_noise_scheduler = get_scheduler(schedule_type='cosine',
                                                            max=max_std,
                                                            hold_steps=self.n_train_itr * self.max_noise_hold_ratio,
                                                            anneal_steps=self.n_train_itr * (1 - self.max_noise_hold_ratio),
                                                            min=max_std_decayed)
            max_explore_noises = [self.explore_noise_scheduler(t) for t in np.arange(self.n_train_itr)]
            min_explore_noises = [min_std for _ in np.arange(self.n_train_itr)]
            plt.figure()
            plt.plot(np.arange(self.n_train_itr), max_explore_noises, label=f'max_std:{max_std:.2f} to {max_std_decayed:.2f}')
            plt.plot(np.arange(self.n_train_itr), min_explore_noises, label=f'min_std:{min_std:.2f}')
            plt.legend()
            name = os.path.join(self.logdir, 'explore_noise') + '.png'
            plt.savefig(name)
            plt.close()
            log.info("Exploration noise level bounds saved to %s" % name)
        else:
            max_std = cfg.model.max_logprob_denoising_std
            min_std = cfg.model.min_logprob_denoising_std
            max_explore_noises = [max_std for _ in np.arange(self.n_train_itr)]
            min_explore_noises = [min_std for _ in np.arange(self.n_train_itr)]
            log.info(f"Received self.model.noise_scheduler_type={self.model.noise_scheduler_type}, will use constant noise ranges [{min_std:.2f}, {max_std:.2f}]")
            plt.figure()
            plt.plot(np.arange(self.n_train_itr), max_explore_noises, label=f'max_std:{max_std:.2f}')
            plt.plot(np.arange(self.n_train_itr), min_explore_noises, label=f'min_std:{min_std:.2f}')
            plt.legend()
            name = os.path.join(self.logdir, 'explore_noise') + '.png'
            plt.savefig(name)
            plt.close()
            log.info("Exploration noise level bounds saved to %s" % name)

        self.denoising_steps = cfg.get('denoising_steps', 1)
        self.verbose = cfg.train.get('verbose', False)
        self.current_best_reward = np.float32('-inf')
        self.is_best_so_far = False
        self.grpo_diagnostics = {}
        self.train_ret_dict = {}
        self.approx_kl = 0.0
        self._group_obs_check_logged = False
        self.initial_ratio_error_threshold = 1e-6  # ratio must be exactly 1.00 at epoch 0 batch 0

        # n_steps is only a safety cap on policy calls per episode rollout:
        # every env ends (terminated or time-limit) within max_episode_steps env steps
        assert (
            self.n_steps * self.act_steps >= self.max_episode_steps
        ), f"rollout cap too small: n_steps*act_steps={self.n_steps * self.act_steps} < max_episode_steps={self.max_episode_steps}"

        self.buffer = GRPOFlowBuffer(
            n_envs=self.n_envs,
            group_size=self.group_size,
            act_steps=self.act_steps,
            best_reward_threshold_for_success=self.best_reward_threshold_for_success,
            furniture_sparse_reward=self.furniture_sparse_reward,
            adv_batch_center=cfg.train.get("adv_batch_center", False),
            adv_positive_only=cfg.train.get("adv_positive_only", False),
            adv_clip=cfg.train.get("adv_clip", None),
            adv_neg_scale=cfg.train.get("adv_neg_scale", 1.0),
            success_from_termination=self.success_from_termination,
        )

    def visualize_lr(self, cfg):
        steps = []
        actor_lrs = []
        # Dry-running the real scheduler would leak n_train_itr steps into training
        # (CosineAnnealingWarmupRestarts has no reset()); snapshot and restore instead.
        sched_state = self.actor_lr_scheduler.state_dict()
        start_lrs = [g["lr"] for g in self.actor_optimizer.param_groups]
        for step in range(cfg.train.n_train_itr):
            self.actor_lr_scheduler.step()
            steps.append(step)
            actor_lrs.append(self.actor_optimizer.param_groups[0]["lr"])
        plt.figure()
        plt.plot(steps, actor_lrs, label='actor', color='blue')
        plt.legend(loc='upper right')
        lr_save_path = os.path.join(self.logdir, 'test_lr_schedulers.png')
        plt.savefig(lr_save_path)
        log.info(f"learning rate saved to {lr_save_path}")
        plt.close()

        self.actor_lr_scheduler.load_state_dict(sched_state)
        for g, lr in zip(self.actor_optimizer.param_groups, start_lrs):
            g["lr"] = lr

        if isinstance(self.actor_lr_scheduler, CustomScheduler):
            self.actor_lr_scheduler.reset()

        self.print_architecture()

    def print_architecture(self):
        arc_path = os.path.join(self.logdir, 'architecture.log')
        with open(arc_path, mode='w') as arc_file:
            arc_file.write(f"self.model=\n{self.model}\nnumber of parameters: {sum([p.numel() for p in self.model.parameters()])/1e6:.2f} M")
        log.info(f"architecture wrote to file {arc_path}")
        arc_file.close()

    ########################################################################
    # Run-loop infrastructure (same patterns as the PPO baseline)
    def prepare_run(self):
        self.timer = Timer()
        self.run_results = []
        self.cnt_train_step = 0
        self.last_itr_eval = False

    def prepare_video_path(self):
        # Video paths for the first episodes of this iteration, if rendering
        self.options_venv = [{} for _ in range(self.n_envs)]
        if self.itr % self.render_freq == 0 and self.render_video:
            for env_ind in range(self.n_render):
                self.options_venv[env_ind]["video_path"] = os.path.join(
                    self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                )

    def set_model_mode(self):
        # Eval on val_freq iterations (and right after resume), train otherwise
        if self.skip_initial_eval and self.itr == 0:
            self.eval_mode = False
        else:
            if self.resume:
                self.eval_mode = True
                self.resume = False
            else:
                self.eval_mode = self.itr % self.val_freq == 0 and not self.force_train
        self.model.eval() if self.eval_mode else self.model.train()
        self.last_itr_eval = self.eval_mode

    ########################################################################
    # GRPO-specific rollout logic
    def grouped_reset_options(self):
        """
        Per-env reset options implementing the GRPO grouped reset. Envs of one
        group receive the SAME seed, different groups different seeds, drawn
        from a base rng seeded by cfg.seed + itr -> runs are reproducible.

        The seed travels through AsyncVectorEnv.reset_arg ->
        MultiStep.reset(options=...) -> RobomimicLowdimWrapper.reset, which
        calls np.random.seed(seed) right before env.reset(); the robosuite
        initial-state randomization consumes the global np.random state, so
        same seed => identical initial state. (TrainAgent relies on the same
        mechanism when it seeds workers at startup.)
        """
        rng = np.random.RandomState(self.seed + self.itr)
        group_seeds = rng.randint(low=0, high=2 ** 31 - 1, size=self.n_groups)
        options_venv = []
        for env_ind in range(self.n_envs):
            opt = dict(self.options_venv[env_ind])  # keep video_path if present
            opt["seed"] = int(group_seeds[env_ind // self.group_size])
            options_venv.append(opt)
        return options_venv

    def check_group_obs_identical(self):
        """Verify the grouped reset produced identical initial obs within each group."""
        group_states = self.prev_obs_venv["state"].reshape(self.n_groups, self.group_size, -1)
        max_diff = float(np.abs(group_states - group_states[:, :1]).max())
        self.grpo_diagnostics["group_init_obs_max_diff"] = max_diff
        if max_diff > 0:
            log.warning(f"Grouped reset is NOT producing identical initial states within groups! max abs diff={max_diff}")
        elif not self._group_obs_check_logged:
            log.info("Grouped reset check passed: initial observations are identical within each group.")
            self._group_obs_check_logged = True

    def rollout_complete_episodes(self):
        """
        Roll every env until its episode ends (done or time-limit). Only steps
        up to and including the terminal step are recorded; envs that finish
        early are masked out afterwards (they keep stepping in the vector env
        for synchrony but their records are closed).
        """
        if self.eval_mode:
            options_venv = self.options_venv  # plain resets, as in the PPO baseline eval
        else:
            options_venv = self.grouped_reset_options()
        self.prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        if not self.eval_mode:
            self.check_group_obs_identical()

        episode_active = np.ones(self.n_envs, dtype=bool)
        for step in range(self.n_steps):
            if not episode_active.any():
                break
            with torch.no_grad():
                cond = {
                    "state": torch.tensor(self.prev_obs_venv["state"], device=self.device, dtype=torch.float32)
                }
                action_samples, chains_venv, logprob_venv = self.get_samples_logprobs(
                    cond=cond,
                    normalize_denoising_horizon=self.normalize_denoising_horizon,
                    normalize_act_space_dimension=self.normalize_act_space_dim,
                    clip_intermediate_actions=self.clip_intermediate_actions,
                    account_for_initial_stochasticity=self.account_for_initial_stochasticity,
                )

            # Apply multi-step action
            action_venv = action_samples[:, : self.act_steps]
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(action_venv)
            done_venv = terminated_venv | truncated_venv

            for env_ind in np.where(episode_active)[0]:
                self.buffer.add(
                    env_ind,
                    self.prev_obs_venv["state"][env_ind],
                    chains_venv[env_ind],
                    logprob_venv[env_ind],
                    reward_venv[env_ind],
                    terminated_venv[env_ind],
                )
                if done_venv[env_ind]:
                    self.buffer.mark_completed(env_ind)
            episode_active[done_venv] = False

            self.prev_obs_venv = obs_venv
            self.cnt_train_step += self.n_envs * self.act_steps if not self.eval_mode else 0

    ########################################################################
    # GRPO update
    def agent_update(self, verbose=True):
        if self.buffer.n_samples == 0:
            log.warning("No valid samples collected this iteration; skipping the GRPO update.")
            self.train_ret_dict = {}
            return

        loss_list, pg_loss_list, entropy_list, ref_reg_list = [], [], [], []
        clipfracs_list, kl_list, ratio_list, noise_std_list = [], [], [], []
        for update_epoch, batch_id, minibatch in self.buffer.minibatch_iterator(
            self.batch_size, self.update_epochs, self.device
        ):
            self.model: GRPOFlow
            pg_loss, entropy_loss, ref_reg, \
            clipfrac, approx_kl, ratio, \
            oldlogprob_min, oldlogprob_max, oldlogprob_std, \
            newlogprob_min, newlogprob_max, newlogprob_std, \
            noise_std = self.model.loss(
                *minibatch,
                use_ref_reg=self.use_ref_reg,
                normalize_denoising_horizon=self.normalize_denoising_horizon,
                normalize_act_space_dimension=self.normalize_act_space_dim,
                verbose=verbose,
                clip_intermediate_actions=self.clip_intermediate_actions,
                account_for_initial_stochasticity=self.account_for_initial_stochasticity,
            )
            self.approx_kl = approx_kl
            if verbose:
                log.info(f"update_epoch={update_epoch}/{self.update_epochs}, batch_id={batch_id}, ratio={ratio:.3f}, clipfrac={clipfrac:.3f}, approx_kl={self.approx_kl:.2e}")

            if update_epoch == 0 and batch_id == 0 and np.abs(ratio - 1.00) > self.initial_ratio_error_threshold:
                raise ValueError(f"ratio={ratio} not 1.00 when update_epoch ==0  and batch_id ==0, there must be some bugs in your code not related to hyperparameters !")

            if self.target_kl and self.lr_schedule == 'adaptive_kl':
                self.update_lr_adaptive_kl(self.approx_kl)

            loss = pg_loss + entropy_loss * self.ent_coef + ref_reg * self.ref_reg_coef

            loss_list.append(loss.item())
            pg_loss_list.append(pg_loss.item())
            entropy_list.append(entropy_loss.item())
            ref_reg_list.append(ref_reg.item())
            clipfracs_list.append(clipfrac)
            kl_list.append(approx_kl)
            ratio_list.append(ratio)
            noise_std_list.append(noise_std)

            # Actor-only gradient step (no critic optimizer exists)
            self.actor_optimizer.zero_grad()
            loss.backward()
            actor_norm = torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), max_norm=float('inf'))
            if verbose:
                log.info(f"before clipping: actor_norm={actor_norm:.2e}")
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.actor_ft.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # Early stop the epoch loop once KL exceeds the target (baseline behavior)
            if self.target_kl and self.lr_schedule == 'fixed' and self.approx_kl > self.target_kl:
                log.warning(f"KL change too much, approx_kl={self.approx_kl} > {self.target_kl}=target_kl, stop optimization.")
                break

        self.train_ret_dict = {
            "loss": float(np.mean(loss_list)),
            "pg_loss": float(np.mean(pg_loss_list)),
            "entropy": float(np.mean(entropy_list)),
            "ref_reg": float(np.mean(ref_reg_list)),
            "approx_kl": float(np.mean(kl_list)),
            "ratio_mean": float(np.mean(ratio_list)),
            "clipfrac": float(np.mean(clipfracs_list)),
            "noise_std": float(np.mean(noise_std_list)),
            "old_logprob_min": oldlogprob_min,
            "old_logprob_max": oldlogprob_max,
            "old_logprob_std": oldlogprob_std,
            "new_logprob_min": newlogprob_min,
            "new_logprob_max": newlogprob_max,
            "new_logprob_std": newlogprob_std,
            "actor_norm": actor_norm,
            "actor_lr": self.actor_optimizer.param_groups[0]["lr"],
            "min_logprob_noise_std": self.model.min_logprob_denoising_std,
            "min_sampling_noise_std": self.model.min_sampling_denoising_std,
        }

    @torch.no_grad()
    def get_samples_logprobs(self,
                             cond: dict,
                             ret_device='cpu',
                             save_chains=True,
                             normalize_denoising_horizon=False,
                             normalize_act_space_dimension=False,
                             clip_intermediate_actions=True,
                             account_for_initial_stochasticity=True):
        # Same sampling path as the PPO flow baseline; action_samples stay numpy for mujoco
        action_samples, chains_venv, logprob_venv = self.model.get_actions(
            cond,
            eval_mode=self.eval_mode,
            save_chains=save_chains,
            normalize_denoising_horizon=normalize_denoising_horizon,
            normalize_act_space_dimension=normalize_act_space_dimension,
            clip_intermediate_actions=clip_intermediate_actions,
            account_for_initial_stochasticity=account_for_initial_stochasticity,
        )
        return (
            action_samples.cpu().numpy(),
            chains_venv.cpu().numpy() if ret_device == 'cpu' else chains_venv,
            logprob_venv.cpu().numpy() if ret_device == 'cpu' else logprob_venv,
        )

    def update_lr(self):
        if self.target_kl and self.lr_schedule == 'adaptive_kl':  # lr adapted per minibatch instead
            return
        self.actor_lr_scheduler.step()
        log.info(f"""learning rate updated. actor_lr={self.actor_optimizer.param_groups[0]["lr"]:.2e}""")

    def update_lr_adaptive_kl(self, approx_kl):
        min_actor_lr = 1e-5
        max_actor_lr = 5e-4
        tune = 'maintains'
        if approx_kl > self.target_kl * 2.0:
            self.actor_lr = max(min_actor_lr, self.actor_lr / 1.5)
            tune = 'decreases'
        elif 0.0 < approx_kl and approx_kl < self.target_kl / 2.0:
            self.actor_lr = min(max_actor_lr, self.actor_lr * 1.5)
            tune = 'increases'
        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = self.actor_lr
        log.info(f"""adaptive kl {tune} lr: actor_lr={self.actor_optimizer.param_groups[0]["lr"]:.2e}""")

    def adjust_finetune_schedule(self):
        # Ported from the PPO flow agent: per-iteration exploration-noise
        # scheduling of NoisyFlowMLP -- orthogonal to the RL algorithm.
        if self.model.noise_scheduler_type == 'const_schedule_itr':
            explore_noise_std = self.explore_noise_scheduler(self.itr)
            self.model.actor_ft.set_logprob_noise_levels(force_level=explore_noise_std)

        # gradually decrease the noise upper bound, to prevent noisy samples from hurting the model
        if self.model.noise_scheduler_type == 'learn_decay':
            updated_noise_std_range = [
                self.model.actor_ft.min_logprob_denoising_std,
                self.explore_noise_scheduler(self.itr),
            ]
            self.model.actor_ft.explore_noise_net.set_noise_range(updated_noise_std_range)
            log.info(f"Updated noise_std_range={updated_noise_std_range} (self.model.noise_scheduler_type={self.model.noise_scheduler_type})")

    ########################################################################
    # Main loop
    def run(self):
        self.prepare_run()
        if self.resume:
            self.resume_training()
        while self.itr < self.n_train_itr:
            self.prepare_video_path()
            self.set_model_mode()
            self.buffer.reset()
            self.rollout_complete_episodes()

            if not self.eval_mode:
                self.buffer.drop_incomplete()
            self.buffer.summarize_episode_reward()

            if not self.eval_mode:
                # merge into existing diagnostics (keeps group_init_obs_max_diff)
                self.grpo_diagnostics.update(
                    self.buffer.compute_returns_and_advantages(
                        gamma=self.gamma,
                        adv_eps=self.adv_eps,
                        use_std_norm=self.adv_use_std_norm,
                    )
                )
                self.agent_update(verbose=self.verbose)

            self.log()
            self.update_lr()
            self.adjust_finetune_schedule()
            self.save_model()
            self.itr += 1

    ########################################################################
    # Checkpointing (actor-only optimizer/scheduler state)
    def save_model(self):
        """
        saves model to disk; no ema recorded because we are doing RLFT.
        "policy" holds only the flow policy network (loadable into a ReFlow
        object's .network for evaluation), without the exploration noise net.
        """
        policy_network_state_dict = {
            'network.' + key: value for key, value in self.model.actor_ft.policy.state_dict().items()
        }
        data = {
            "itr": self.itr,
            "cnt_train_steps": self.cnt_train_step,
            "model": self.model.state_dict(),  # for resume training
            "policy": policy_network_state_dict,  # flow policy for evaluation
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_scheduler": self.actor_lr_scheduler.state_dict(),
        }

        # always save the last model for resume of training
        save_path = os.path.join(self.checkpoint_dir, f"last.pt")
        torch.save(data, os.path.join(self.checkpoint_dir, save_path))

        # optionally save intermediate models
        if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
            save_path = os.path.join(self.checkpoint_dir, f"state_{self.itr}.pt")
            torch.save(data, os.path.join(self.checkpoint_dir, save_path))
            log.info(f"\n Saved model at itr={self.itr} to {save_path}\n ")

        # save the best model evaluated so far
        if self.is_best_so_far:
            save_path = os.path.join(self.checkpoint_dir, f"best.pt")
            torch.save(data, os.path.join(self.checkpoint_dir, save_path))
            log.info(f"\n Saved model with the highest evaluated average episode reward {self.current_best_reward:4.3f} to \n{save_path}\n ")
            self.is_best_so_far = False

    def resume_training(self):
        log.info(f"Resuming training...")
        data = torch.load(self.resume_path, weights_only=True, map_location=self.device)
        self.itr = data["itr"]
        self.cnt_train_step = (
            self.itr * self.n_envs * self.act_steps * self.n_steps
            if 'cnt_train_steps' not in data.keys()
            else data["cnt_train_steps"]
        )
        self.n_train_itr += self.itr  # train for another xx iters
        log.info(f"Resume training from itr={self.itr}, total train steps={self.cnt_train_step}.")

        if "model" in data.keys():
            self.model.load_state_dict(data["model"], strict=True)
            log.info("Loaded full model.")
        elif "policy" in data.keys():
            self.model.actor_ft.policy.load_state_dict(data["policy"], strict=True)
            log.info("Loaded policy. Initialize exploration noise network from scratch.")
        else:
            raise ValueError("Your saved checkpoint does not contain keys like 'model' and 'policy'. Please check what was wrong with your model saving functions.")
        log.info(f"Successfully loaded model from path={self.resume_path}")

        self.actor_optimizer.load_state_dict(data["actor_optimizer"])
        log.info(f"Successfully loaded optimizer from path={self.resume_path}")

        if 'actor_lr_scheduler' in data.keys():
            self.actor_lr_scheduler.load_state_dict(data["actor_lr_scheduler"])
            log.info(f"Successfully loaded scheduler from path={self.resume_path}")
        else:
            for _ in range(self.itr):  # recover lr scheduler
                self.actor_lr_scheduler.step()
            log.info(f"No scheduler found in path={self.resume_path}. Automatically calibrate the newly initialized scheduler.")

        if self.model.noise_scheduler_type == 'const':
            updated_noise_std_range = [
                self.cfg.model.min_logprob_denoising_std,
                self.cfg.model.max_logprob_denoising_std,
            ]
            self.model.actor_ft.explore_noise_net.set_noise_range(updated_noise_std_range)
            log.info(f"Updated noise_std_range={updated_noise_std_range} (self.model.noise_scheduler_type={self.model.noise_scheduler_type})")

    ########################################################################
    # Logging: same metric names as the PPO baseline so curves overlay,
    # plus GRPO-specific diagnostics under grpo/
    def log(self):
        BOLDSTART = '\033[1m'
        BOLDEND = '\033[0m'

        self.run_results.append(
            {
                "itr": self.itr,
                "step": self.cnt_train_step,
            }
        )
        if self.itr % self.log_freq == 0:
            time = self.timer()
            self.run_results[-1]["time"] = time
            if self.eval_mode:
                log.info(create_bordered_text(
                    f"{BOLDSTART}Evaluation at itr {self.itr}{BOLDEND}:\n"
                    f"Model: {self.model.__class__.__name__}\n"
                    f"Environment: {self.env_name} x {self.n_envs}\n"
                    f"Num denoising steps: {self.denoising_steps}\n"
                    f"Seed: {self.seed}\n"
                    f"Success Rate: {self.buffer.success_rate * 100:3.2f}% ± {self.buffer.std_success_rate * 100:3.2f}%\n"
                    f"Episode Reward: {self.buffer.avg_episode_reward:8.2f} ± {self.buffer.std_episode_reward:8.2f}\n"
                    f"Best Reward (per action): {self.buffer.avg_best_reward:8.2f} ± {self.buffer.std_best_reward:8.2f}\n"
                    f"Episode Length: {self.buffer.avg_episode_length:8.2f} ± {self.buffer.std_episode_length:8.2f}\n"
                    f"Actor lr :{self.actor_optimizer.param_groups[0]['lr']:.2e}"
                ))
                eval_dict = {
                    "eval/success rate": self.buffer.success_rate,
                    "eval/avg episode reward": self.buffer.avg_episode_reward,
                    "eval/avg best reward": self.buffer.avg_best_reward,
                    "eval/avg episode length": self.buffer.avg_episode_length,
                    "eval/num episode": self.buffer.num_episode_finished,
                    "eval/std success rate": self.buffer.std_success_rate,
                    "eval/std episode reward": self.buffer.std_episode_reward,
                    "eval/std best reward": self.buffer.std_best_reward,
                    "eval/std episode length": self.buffer.std_episode_length,
                }
                for key, value in eval_dict.items():
                    if isinstance(value, torch.Tensor):
                        eval_dict[key] = value.item()
                self.run_results[-1].update(eval_dict)
                if self.use_wandb:
                    wandb.log(
                        data=eval_dict,
                        step=self.itr,
                        commit=False,
                    )

                if self.current_best_reward < self.buffer.avg_episode_reward:
                    self.current_best_reward = self.buffer.avg_episode_reward
                    self.is_best_so_far = True
                    log.info(f"New best reward evaluated: {self.current_best_reward:4.3f}")
            else:
                train_prt_str_basic = (
                    f"itr {self.itr} | Total Step {self.cnt_train_step / 1e6:4.3f} M | Time: {time:8.3f}\n"
                    f"Env: {self.env_name} x {self.n_envs}\n"
                    f"Episode Reward: {self.buffer.avg_episode_reward:8.2f} ± {self.buffer.std_episode_reward:8.2f}\n"
                    f"Success Rate: {self.buffer.success_rate * 100:3.2f}% ± {self.buffer.std_success_rate * 100:3.2f}% \n"
                    f"Avg Best Reward: {self.buffer.avg_best_reward:8.2f} ± {self.buffer.std_best_reward:8.2f}\n"
                    f"Episode Length: {self.buffer.avg_episode_length:8.2f} ± {self.buffer.std_episode_length:8.2f}\n"
                    f"Effective Group Ratio: {self.grpo_diagnostics.get('effective_group_ratio', 0.0):4.2f}\n"
                    f"Adv Clip Fraction: {self.grpo_diagnostics.get('adv_clip_frac', 0.0):4.2f}\n"
                    f"Actor lr :{self.actor_optimizer.param_groups[0]['lr']:.2e}\n"
                )
                formatted_items = [f"{key}: {value:.3e}" for key, value in self.train_ret_dict.items()]
                num_items_per_row = 10
                for i in range(0, len(formatted_items), num_items_per_row):
                    train_prt_str_basic += " | ".join(formatted_items[i:i+num_items_per_row]) + "\n"
                log.info(train_prt_str_basic)

                train_log_dict_basic = {
                    "train/total env step": self.cnt_train_step,
                    "train/success rate": self.buffer.success_rate,
                    "train/avg episode reward": self.buffer.avg_episode_reward,
                    "train/avg episode length": self.buffer.avg_episode_length,
                    "train/num episode": self.buffer.num_episode_finished,
                    "train/std success rate": self.buffer.std_success_rate,
                    "train/avg best reward": self.buffer.avg_best_reward,
                    "train/std episode reward": self.buffer.std_episode_reward,
                    "train/std best reward": self.buffer.std_best_reward,
                    "train/std episode length": self.buffer.std_episode_length,
                    "train/actor lr": self.actor_optimizer.param_groups[0]["lr"],
                }
                grpo_dict = {"grpo/" + key: value for key, value in self.grpo_diagnostics.items()}
                train_log_dict_basic.update(grpo_dict)
                loss_dict = {"loss/" + key: value for key, value in self.train_ret_dict.items()}
                train_log_dict_basic.update(loss_dict)
                for key, value in train_log_dict_basic.items():
                    if isinstance(value, torch.Tensor):
                        train_log_dict_basic[key] = value.item()
                self.run_results[-1].update(train_log_dict_basic)

                if self.use_wandb:
                    wandb.log(
                        data=train_log_dict_basic,
                        step=self.itr,
                        commit=True,
                    )
            with open(self.result_path, "wb") as f:
                pickle.dump(self.run_results, f)
