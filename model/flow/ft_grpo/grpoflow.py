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
GRPOFlow: flow-matching policy model for GRPO fine-tuning -- the critic-free
counterpart of PPOFlow.

GRPO mapping (initial state = prompt, group of episodes = completions,
trajectory return = outcome reward): the baseline in the policy-gradient
objective is the MEAN RETURN OF THE GROUP of episodes that share the same
initial state, not a learned value function. This model therefore holds ONLY:
    actor_ft  : NoisyFlowMLP -- fine-tuned policy with exploration noise net
    actor_old : frozen FlowMLP -- SFT base policy = GRPO's reference policy
No critic is built, stored, or referenced anywhere (optimizer, loss, buffer).

The objective follows the GRPO paper: the same PPO-style clipped
importance-sampling surrogate over denoising-chain logprob ratios. Chain
logprobs are recomputed with the inherited PPOFlow.get_logprobs -- that
machinery is algorithm-shared (it knows nothing about critics or GAE).
Differences from PPOFlow.loss: no value loss, and advantages are NOT
re-normalized per batch -- they are already group-relative, and re-centering
them per minibatch would partially undo the group structure that GRPO relies on.
"""
import torch
import torch.nn.functional as F
import logging
log = logging.getLogger(__name__)
from model.flow.ft_ppo.ppoflow import PPOFlow


class GRPOFlow(PPOFlow):
    def __init__(self, critic=None, **kwargs):
        # GRPO is critic-free: a critic config is accepted for signature
        # compatibility with PPOFlow and explicitly discarded.
        if critic is not None:
            log.warning("GRPOFlow received a critic but ignores it: GRPO is critic-free.")
        super().__init__(critic=None, **kwargs)
        assert self.critic is None, "GRPOFlow must not hold a critic."

    def loss(
        self,
        obs,
        chains,
        advantages,
        oldlogprobs,
        use_ref_reg=False,
        normalize_denoising_horizon=False,
        normalize_act_space_dimension=False,
        verbose=True,
        clip_intermediate_actions=True,
        account_for_initial_stochasticity=True,
    ):
        """
        GRPO loss = clipped surrogate over chain-logprob ratios + entropy bonus
        (+ optional regularization toward the frozen reference policy).

        obs: dict, "state": (B, To, Do)
        chains: (B, K+1, Ta, Da) full denoising chains from the old policy
        advantages: (B,) group-relative advantages, broadcast per episode step.
            Unlike PPOFlow.loss they are NOT re-normalized per batch.
        oldlogprobs: (B,) chain logprobs under the rollout (old) policy
        """
        newlogprobs, entropy, noise_std = self.get_logprobs(
            obs,
            chains,
            get_entropy=True,
            normalize_denoising_horizon=normalize_denoising_horizon,
            normalize_act_space_dimension=normalize_act_space_dimension,
            verbose_entropy_stats=verbose,
            clip_intermediate_actions=clip_intermediate_actions,
            account_for_initial_stochasticity=account_for_initial_stochasticity,
        )
        if verbose:
            log.info(f"oldlogprobs.min={oldlogprobs.min():5.3f}, max={oldlogprobs.max():5.3f}, std of oldlogprobs={oldlogprobs.std():5.3f}")
            log.info(f"newlogprobs.min={newlogprobs.min():5.3f}, max={newlogprobs.max():5.3f}, std of newlogprobs={newlogprobs.std():5.3f}")

        newlogprobs = newlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)
        oldlogprobs = oldlogprobs.clamp(min=self.logprob_min, max=self.logprob_max)
        if verbose:
            if oldlogprobs.min() < self.logprob_min: log.info(f"WARNINIG: old logprobs too low, potential policy collapse detected, should encourage exploration.")
            if newlogprobs.min() < self.logprob_min: log.info(f"WARNINIG: new logprobs too low, potential policy collapse detected, should encourage exploration.")
            if newlogprobs.max() > self.logprob_max: log.info(f"WARNINIG: new logprobs too high")
            if oldlogprobs.max() > self.logprob_max: log.info(f"WARNINIG: old logprobs too high")

        # Importance-sampling ratio between current and rollout policy
        logratio = newlogprobs - oldlogprobs
        ratio = logratio.exp()

        # KL estimate and clip fraction (same estimator style as the PPO baseline)
        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > self.clip_ploss_coef).float().mean().item()

        # Clipped surrogate objective (GRPO paper uses the PPO-style clip)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_ploss_coef, 1 + self.clip_ploss_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Entropy bonus
        entropy_loss = -entropy.mean()
        if verbose:
            with torch.no_grad():
                log.info(f"Entropy Percentiles: 10%={entropy.quantile(0.1):.2f}, 50%={entropy.median():.2f}, 90%={entropy.quantile(0.9):.2f}")

        # Optional regularization toward the frozen reference policy actor_old:
        # action-space (W2-style) distance between reference and fine-tuned flow
        # under the SAME recorded initial noise z = chains[:, 0]. Default OFF.
        ref_reg = torch.zeros((), device=self.device)
        if use_ref_reg:
            z = chains[:, 0]
            a_ref = self.actor_old.sample_action(cond=obs, inference_steps=self.inference_steps, clip_intermediate_actions=True, act_range=[self.act_min, self.act_max], z=z)
            a_theta = self.actor_ft.policy.sample_action(cond=obs, inference_steps=self.inference_steps, clip_intermediate_actions=True, act_range=[self.act_min, self.act_max], z=z)
            ref_reg = F.mse_loss(a_ref.detach(), a_theta)

        return (
            pg_loss,
            entropy_loss,
            ref_reg,
            clipfrac,
            approx_kl.item(),
            ratio.mean().item(),
            oldlogprobs.min(),
            oldlogprobs.max(),
            oldlogprobs.std(),
            newlogprobs.min(),
            newlogprobs.max(),
            newlogprobs.std(),
            noise_std.item(),
        )
