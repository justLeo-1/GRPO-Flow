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
Episode-organized rollout buffer for GRPO (Group Relative Policy Optimization).

Why not PPOBuffer: the PPO buffers are fixed-size (n_steps x n_envs) arrays that
store fixed-length, possibly truncated rollout segments and rely on a critic +
GAE to bootstrap values at segment boundaries. GRPO is critic-free: every
rollout is a COMPLETE episode (terminated or time-limit truncated), and the
advantage of an episode is its return compared against the returns of the other
episodes in its initial-state group. The natural container is therefore a
variable-length list of per-episode step records with group ids attached.
"""
import numpy as np
import torch
import logging
log = logging.getLogger(__name__)


class GRPOFlowBuffer:
    """
    List-based buffer of complete episodes for flow-policy GRPO fine-tuning.

    Layout: env i belongs to group (i // group_size); envs of one group were
    reset to the SAME initial state (= the GRPO "prompt"), so their episodes
    (= "completions") are comparable. Each step record stores everything the
    update needs to rebuild the importance-sampling ratio:
        state   (n_cond_step, obs_dim)          policy input at this call
        chain   (inference_steps+1, Ta, Da)     full denoising chain sampled by the old policy
        logprob ()                              chain logprob under the old policy
        reward  ()                              reward of the executed action chunk
        group_id ()                             initial-state group of this episode
    """

    def __init__(self,
                 n_envs,
                 group_size,
                 act_steps,
                 best_reward_threshold_for_success,
                 furniture_sparse_reward=False,
                 adv_batch_center=False,
                 adv_positive_only=False,
                 adv_clip=None,
                 adv_neg_scale=1.0,
                 success_from_termination=False):
        assert n_envs % group_size == 0, (
            f"n_envs={n_envs} must be divisible by group_size={group_size}"
        )
        self.n_envs = n_envs
        self.group_size = group_size
        self.n_groups = n_envs // group_size
        self.act_steps = act_steps
        self.best_reward_threshold_for_success = best_reward_threshold_for_success
        self.furniture_sparse_reward = furniture_sparse_reward
        # Per-batch centering of the per-step advantages. Group-relative centering
        # zeroes the mean per EPISODE, but failed episodes are LONGER (time limit)
        # than successful ones (early truncation), so the step-weighted batch mean
        # is negative -> a constant net downward pressure on all sampled actions.
        # Subtracting the batch mean removes this mechanical bias while keeping
        # the group-relative structure intact.
        self.adv_batch_center = adv_batch_center
        # Positive-only advantages (RWR-style): failed trajectories contribute no
        # gradient at all, cutting off the blanket-punishment channel where the
        # correct prefix of a failed episode gets suppressed. Mutually exclusive
        # with batch centering (centering would reintroduce negative mass).
        self.adv_positive_only = adv_positive_only
        # Symmetric advantage clipping (z-score units, since advantages are
        # std-normalized within groups). At ~60% success the minority failures
        # carry |A|~1.29 vs +0.78 for successes, and singleton outlier groups
        # reach +/-3.87 (G=16 binary). c=1.0 softens the failure punishment by
        # ~22% in typical mixed groups and kills the outlier spikes entirely --
        # a middle point between signed advantages and positive_only.
        self.adv_clip = adv_clip
        # Negative-advantage scaling: A <- A * adv_neg_scale for A < 0, applied
        # AFTER group normalization and BEFORE clipping. Rationale: at ~60-70%
        # success the minority failures get larger-magnitude z-scores than the
        # successes, so the gradient budget tilts toward punishing failed
        # trajectories (whose prefixes are often correct). Scaling shifts budget
        # from punishment toward reward while preserving the relative ordering
        # among failures (unlike clipping, which flattens). 1.0 = no-op.
        self.adv_neg_scale = adv_neg_scale
        # if True, episode success is read from the termination flag of the
        # episode's final step instead of the reward threshold (robust to dense
        # staged rewards, where partial credit inflates episode_best_reward)
        self.success_from_termination = success_from_termination
        self.reset()

    def reset(self):
        self.episodes = [[] for _ in range(self.n_envs)]
        self.completed = np.zeros(self.n_envs, dtype=bool)
        self.returns = np.full(self.n_envs, np.nan)
        self.advantages = np.zeros(self.n_envs)
        self.diagnostics = {}

    @property
    def n_samples(self):
        return sum(len(ep) for ep in self.episodes)

    def add(self, env_ind, state, chain, logprob, reward, terminated):
        """Append one policy-call record; call only while env_ind's episode is active."""
        self.episodes[env_ind].append(
            {
                "state": state.copy(),
                "chain": chain.copy(),
                "logprob": float(logprob),
                "reward": float(reward),
                "terminated": bool(terminated),
                "group_id": env_ind // self.group_size,
            }
        )

    def mark_completed(self, env_ind):
        self.completed[env_ind] = True

    def drop_incomplete(self):
        """
        GRPO outcome supervision requires complete episodes (terminated or
        time-limit truncated). Episodes still open when the rollout loop ended
        have no well-defined outcome return and are dropped.
        """
        n_dropped = 0
        for env_ind in range(self.n_envs):
            if not self.completed[env_ind] and len(self.episodes[env_ind]) > 0:
                n_dropped += 1
                self.episodes[env_ind] = []
        if n_dropped > 0:
            log.warning(
                f"Dropped {n_dropped} incomplete episodes (no episode end within the rollout cap)."
            )
        return n_dropped

    def compute_returns_and_advantages(self, gamma, adv_eps, use_std_norm):
        """
        Outcome return per episode: R_i = sum_t gamma^t * r_it  (no bootstrap --
        there is no critic). Group-relative advantage:
            A_i = (R_i - mean_g) / (std_g + adv_eps)   if use_std_norm
            A_i =  R_i - mean_g                        otherwise
        broadcast to every recorded step of episode i. Groups whose returns have
        (near-)zero variance yield zero advantage -- no learning signal -- so we
        report the fraction of non-degenerate ("effective") groups.
        """
        returns = np.full(self.n_envs, np.nan)
        for env_ind in range(self.n_envs):
            ep = self.episodes[env_ind]
            if len(ep) == 0:
                continue
            rewards = np.array([rec["reward"] for rec in ep])
            discounts = gamma ** np.arange(len(ep))
            returns[env_ind] = np.sum(discounts * rewards)
        self.returns = returns

        advantages = np.zeros(self.n_envs)
        n_effective_groups = 0
        n_valid_groups = 0
        n_valid_envs = 0
        n_clipped = 0
        for g in range(self.n_groups):
            inds = [g * self.group_size + j for j in range(self.group_size)]
            group_returns = np.array([returns[i] for i in inds])
            valid = ~np.isnan(group_returns)
            if valid.sum() == 0:
                continue
            n_valid_groups += 1
            n_valid_envs += int(valid.sum())
            r_g = group_returns[valid]
            mean_g = r_g.mean()
            std_g = r_g.std()  # population std (ddof=0), as in GRPO
            if std_g > 1e-8:
                n_effective_groups += 1
            if use_std_norm:
                adv_g = (group_returns - mean_g) / (std_g + adv_eps)
            else:
                adv_g = group_returns - mean_g
            if self.adv_neg_scale != 1.0:
                adv_g = np.where(adv_g < 0, adv_g * self.adv_neg_scale, adv_g)
            if self.adv_positive_only:
                adv_g = np.clip(adv_g, 0.0, None)
            if self.adv_clip is not None:
                n_clipped += int((np.abs(adv_g[valid]) > self.adv_clip).sum())
                adv_g = np.clip(adv_g, -self.adv_clip, self.adv_clip)
            adv_g[~valid] = 0.0
            for i, a in zip(inds, adv_g):
                advantages[i] = a
        self.advantages = advantages

        valid_returns = returns[~np.isnan(returns)]
        ep_lengths = np.array([len(ep) for ep in self.episodes if len(ep) > 0])
        self.diagnostics = {
            "effective_group_ratio": n_effective_groups / max(1, n_valid_groups),
            "return_mean": float(valid_returns.mean()) if len(valid_returns) > 0 else 0.0,
            "return_std": float(valid_returns.std()) if len(valid_returns) > 0 else 0.0,
            "episode_length_mean": float(ep_lengths.mean() * self.act_steps) if len(ep_lengths) > 0 else 0.0,
            "n_samples": int(ep_lengths.sum()) if len(ep_lengths) > 0 else 0,
            "n_episodes": int(len(ep_lengths)),
            "adv_clip_frac": n_clipped / max(1, n_valid_envs),
        }
        return self.diagnostics

    def make_dataset(self, device):
        """Flatten all recorded steps into tensors (advantage already broadcast per step)."""
        states, chains, advantages, logprobs = [], [], [], []
        for env_ind in range(self.n_envs):
            adv = self.advantages[env_ind]
            for rec in self.episodes[env_ind]:
                states.append(rec["state"])
                chains.append(rec["chain"])
                advantages.append(adv)
                logprobs.append(rec["logprob"])
        obs = torch.tensor(np.stack(states), device=device).float()
        chains = torch.tensor(np.stack(chains), device=device).float()
        advantages = torch.tensor(np.array(advantages), device=device).float()
        if self.adv_batch_center and not self.adv_positive_only:
            advantages = advantages - advantages.mean()
        logprobs = torch.tensor(np.array(logprobs), device=device).float()
        return obs, chains, advantages, logprobs

    def minibatch_iterator(self, batch_size, update_epochs, device):
        """Yield (update_epoch, batch_id, minibatch) over the flattened dataset."""
        obs, chains, advantages, oldlogprobs = self.make_dataset(device)
        total_steps = obs.shape[0]
        for update_epoch in range(update_epochs):
            indices = torch.randperm(total_steps, device=device)
            for batch_id, start in enumerate(range(0, total_steps, batch_size)):
                inds_b = indices[start : start + batch_size]
                minibatch = (
                    {"state": obs[inds_b]},
                    chains[inds_b],
                    advantages[inds_b],
                    oldlogprobs[inds_b],
                )
                yield update_epoch, batch_id, minibatch

    def summarize_episode_reward(self):
        """
        Episode statistics for logging. Metric definitions are identical to
        PPOBuffer.summarize_episode_reward so GRPO and PPO curves overlay.
        """
        episodes = [ep for ep in self.episodes if len(ep) > 0]
        reward_trajs_split = [
            np.array([rec["reward"] for rec in ep])
            for ep in episodes
        ]
        if len(reward_trajs_split) > 0:
            self.num_episode_finished = len(reward_trajs_split)
            episode_reward = np.array(
                [np.sum(reward_traj) for reward_traj in reward_trajs_split]
            )
            if self.furniture_sparse_reward:
                episode_best_reward = episode_reward
            else:
                episode_best_reward = np.array(
                    [
                        np.max(reward_traj) / self.act_steps
                        for reward_traj in reward_trajs_split
                    ]
                )
            self.avg_episode_reward = np.mean(episode_reward)
            self.avg_best_reward = np.mean(episode_best_reward)
            if self.success_from_termination:
                # every recorded episode is complete (drop_incomplete ran first),
                # so its last step's terminated flag tells whether it ended by
                # task success (terminated=True <=> success for robomimic here)
                episode_success = np.array(
                    [ep[-1]["terminated"] for ep in episodes], dtype=bool
                )
            else:
                episode_success = (
                    episode_best_reward >= self.best_reward_threshold_for_success
                )
            self.success_rate = np.mean(episode_success)

            self.std_episode_reward = np.std(episode_reward)
            self.std_best_reward = np.std(episode_best_reward)
            self.std_success_rate = np.std(episode_success)

            episode_lengths = (
                np.array([len(reward_traj) for reward_traj in reward_trajs_split])
                * self.act_steps
            )
            self.avg_episode_length = np.mean(episode_lengths)
            self.std_episode_length = np.std(episode_lengths)
        else:
            self.num_episode_finished = 0
            self.avg_episode_reward = 0
            self.avg_best_reward = 0
            self.success_rate = 0
            self.avg_episode_length = 0.0
            self.std_episode_reward = 0
            self.std_best_reward = 0
            self.std_success_rate = 0
            self.std_episode_length = 0.0
            log.info("[WARNING] No episode completed within the iteration!")
