# --------------------------------------------------------
# SAC (Soft Actor-Critic) for High-Precision Insertion
# Drop-in replacement for frozen_ppo.py
# --------------------------------------------------------
# Key advantages over PPO for tight-clearance insertion:
#   1. Off-policy → ~5-10x sample efficiency via replay buffer
#   2. Entropy-regularized → automatic exploration scheduling
#   3. Learned state-dependent variance → precision when needed
# --------------------------------------------------------

import os
import time
import copy
import cv2
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from termcolor import cprint
from collections import deque

from isaacgyminsertion.tasks.factory_tactile.factory_utils import RotationTransformer
from algo.models.running_mean_std import RunningMeanStd
from isaacgyminsertion.utils.misc import AverageScalarMeter
from tensorboardX import SummaryWriter

import yaml
import json

# ============================================================
# 1. Replay Buffer (GPU-resident for IsaacGym speed)
# ============================================================

class ReplayBuffer:
    """GPU-resident replay buffer optimized for IsaacGym's vectorized envs."""

    def __init__(self, capacity, obs_dim, action_dim, priv_dim, num_envs, device):
        self.capacity = capacity
        self.device = device
        self.num_envs = num_envs
        self.ptr = 0
        self.size = 0

        # Pre-allocate all tensors on GPU
        self.obs = torch.zeros(capacity, obs_dim, device=device)
        self.next_obs = torch.zeros(capacity, obs_dim, device=device)
        self.actions = torch.zeros(capacity, action_dim, device=device)
        self.rewards = torch.zeros(capacity, 1, device=device)
        self.dones = torch.zeros(capacity, 1, device=device)
        self.priv_info = torch.zeros(capacity, priv_dim, device=device)
        self.next_priv_info = torch.zeros(capacity, priv_dim, device=device)

    def add_batch(self, obs, next_obs, actions, rewards, dones, priv_info, next_priv_info):
        """Add a batch of transitions (from all envs simultaneously)."""
        batch_size = obs.shape[0]

        if self.ptr + batch_size > self.capacity:
            # Wrap around
            overflow = (self.ptr + batch_size) - self.capacity
            remaining = batch_size - overflow

            self.obs[self.ptr:self.ptr + remaining] = obs[:remaining]
            self.next_obs[self.ptr:self.ptr + remaining] = next_obs[:remaining]
            self.actions[self.ptr:self.ptr + remaining] = actions[:remaining]
            self.rewards[self.ptr:self.ptr + remaining] = rewards[:remaining]
            self.dones[self.ptr:self.ptr + remaining] = dones[:remaining]
            self.priv_info[self.ptr:self.ptr + remaining] = priv_info[:remaining]
            self.next_priv_info[self.ptr:self.ptr + remaining] = next_priv_info[:remaining]

            self.obs[:overflow] = obs[remaining:]
            self.next_obs[:overflow] = next_obs[remaining:]
            self.actions[:overflow] = actions[remaining:]
            self.rewards[:overflow] = rewards[remaining:]
            self.dones[:overflow] = dones[remaining:]
            self.priv_info[:overflow] = priv_info[remaining:]
            self.next_priv_info[:overflow] = next_priv_info[remaining:]

            self.ptr = overflow
        else:
            self.obs[self.ptr:self.ptr + batch_size] = obs
            self.next_obs[self.ptr:self.ptr + batch_size] = next_obs
            self.actions[self.ptr:self.ptr + batch_size] = actions
            self.rewards[self.ptr:self.ptr + batch_size] = rewards
            self.dones[self.ptr:self.ptr + batch_size] = dones
            self.priv_info[self.ptr:self.ptr + batch_size] = priv_info
            self.next_priv_info[self.ptr:self.ptr + batch_size] = next_priv_info
            self.ptr = (self.ptr + batch_size) % self.capacity

        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        """Uniformly sample a batch of transitions."""
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            'obs': self.obs[idx],
            'next_obs': self.next_obs[idx],
            'actions': self.actions[idx],
            'rewards': self.rewards[idx],
            'dones': self.dones[idx],
            'priv_info': self.priv_info[idx],
            'next_priv_info': self.next_priv_info[idx],
        }


# ============================================================
# 2. Actor Network (Squashed Gaussian Policy)
# ============================================================

LOG_STD_MIN = -10
LOG_STD_MAX = 2

class SquashedGaussianActor(nn.Module):
    """
    Policy that outputs a squashed (tanh) Gaussian.
    Key difference from PPO: learns state-dependent log_std,
    so it can output tiny variance for precise actions.
    """

    def __init__(self, obs_dim, action_dim, hidden_units, priv_info_dim, priv_mlp_units):
        super().__init__()

        # Privileged info encoder (matches your PPO teacher)
        priv_layers = []
        prev_dim = priv_info_dim
        for u in priv_mlp_units:
            priv_layers.extend([nn.Linear(prev_dim, u), nn.ELU()])
            prev_dim = u
        self.priv_encoder = nn.Sequential(*priv_layers)
        priv_embed_dim = priv_mlp_units[-1]

        # Main policy network
        total_input = obs_dim + priv_embed_dim
        layers = []
        prev_dim = total_input
        for u in hidden_units:
            layers.extend([nn.Linear(prev_dim, u), nn.ELU()])
            prev_dim = u

        self.trunk = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)

        # Initialize output layers with small weights for stable start
        self.mu_head.weight.data.mul_(0.1)
        self.mu_head.bias.data.zero_()
        self.log_std_head.weight.data.mul_(0.1)
        self.log_std_head.bias.data.zero_()

    def forward(self, obs, priv_info):
        priv_embed = self.priv_encoder(priv_info)
        x = torch.cat([obs, priv_embed], dim=-1)
        x = self.trunk(x)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs, priv_info):
        """Sample action with reparameterization trick."""
        mu, log_std = self.forward(obs, priv_info)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        # Reparameterization trick
        x_t = dist.rsample()
        action = torch.tanh(x_t)

        # Log prob with tanh correction
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, mu

    def act_deterministic(self, obs, priv_info):
        """Deterministic action for evaluation."""
        mu, _ = self.forward(obs, priv_info)
        return torch.tanh(mu)


# ============================================================
# 3. Critic Network (Twin Q-functions for stability)
# ============================================================

class TwinQCritic(nn.Module):
    """
    Twin Q-networks to mitigate overestimation bias.
    Each Q-network: Q(s, a, priv) -> scalar
    """

    def __init__(self, obs_dim, action_dim, hidden_units, priv_info_dim, priv_mlp_units):
        super().__init__()

        # Shared priv encoder architecture (but separate weights)
        def make_priv_encoder():
            layers = []
            prev_dim = priv_info_dim
            for u in priv_mlp_units:
                layers.extend([nn.Linear(prev_dim, u), nn.ELU()])
                prev_dim = u
            return nn.Sequential(*layers)

        def make_q_net(input_dim):
            layers = []
            prev_dim = input_dim
            for u in hidden_units:
                layers.extend([nn.Linear(prev_dim, u), nn.ELU()])
                prev_dim = u
            layers.append(nn.Linear(prev_dim, 1))
            return nn.Sequential(*layers)

        priv_embed_dim = priv_mlp_units[-1]
        q_input_dim = obs_dim + action_dim + priv_embed_dim

        self.priv_encoder1 = make_priv_encoder()
        self.priv_encoder2 = make_priv_encoder()
        self.q1 = make_q_net(q_input_dim)
        self.q2 = make_q_net(q_input_dim)

    def forward(self, obs, action, priv_info):
        priv1 = self.priv_encoder1(priv_info)
        priv2 = self.priv_encoder2(priv_info)
        x1 = torch.cat([obs, action, priv1], dim=-1)
        x2 = torch.cat([obs, action, priv2], dim=-1)
        return self.q1(x1), self.q2(x2)

    def q1_forward(self, obs, action, priv_info):
        priv1 = self.priv_encoder1(priv_info)
        x1 = torch.cat([obs, action, priv1], dim=-1)
        return self.q1(x1)


# ============================================================
# 4. SAC Agent (drop-in replacement for PPO class)
# ============================================================

class SAC(object):
    """
    SAC agent that mirrors the PPO interface from frozen_ppo.py.
    
    Your train.py calls:  agent = eval(cfg.train.algo)(envs, output_dif, full_config=cfg)
    So this class must accept the same constructor signature.
    """

    def __init__(self, env, output_dif, full_config):

        # ---- MultiGPU (same as PPO) ----
        self.multi_gpu = full_config.train.ppo.multi_gpu
        if self.multi_gpu:
            self.rank = int(os.getenv("LOCAL_RANK", "0"))
            self.rank_size = int(os.getenv("WORLD_SIZE", "1"))
            dist.init_process_group("nccl", rank=self.rank, world_size=self.rank_size)
            self.device = "cuda:" + str(self.rank)
        else:
            self.rank = -1
            self.device = full_config["rl_device"]

        self.full_config = full_config
        self.task_config = full_config.task
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo  # reuse same config section

        # ---- Environment ----
        self.env = env
        self.num_actors = self.ppo_config['num_actors']
        self.actions_num = self.task_config.env.numActions
        self.obs_shape = (self.task_config.env.numObservations * self.task_config.env.numObsHist,)

        # ---- Priv Info ----
        self.priv_info = self.ppo_config['priv_info']
        self.priv_info_dim = self.ppo_config['priv_info_dim']
        self.gt_contacts_info = self.ppo_config['compute_contact_gt']

        # ---- SAC Hyperparameters ----
        # These can be added to your YAML or use sensible defaults
        sac_cfg = full_config.train.get('sac', {})
        self.gamma = sac_cfg.get('gamma', 0.99)
        self.tau_soft = sac_cfg.get('tau', 0.005)  # soft target update
        self.actor_lr = sac_cfg.get('actor_lr', 3e-4)
        self.critic_lr = sac_cfg.get('critic_lr', 3e-4)
        self.alpha_lr = sac_cfg.get('alpha_lr', 3e-4)
        self.batch_size_sac = sac_cfg.get('batch_size', 4096)
        self.buffer_size = sac_cfg.get('buffer_size', 1_000_000)
        self.warmup_steps = sac_cfg.get('warmup_steps', 10000)
        self.updates_per_step = sac_cfg.get('updates_per_step', 1)
        self.init_temperature = sac_cfg.get('init_temperature', 0.2)
        self.learnable_temperature = sac_cfg.get('learnable_temperature', True)
        self.normalize_input = self.ppo_config.get('normalize_input', True)

        # ---- Build Networks ----
        obs_dim = self.obs_shape[0]
        action_dim = self.actions_num
        hidden_units = list(self.network_config.mlp.units)
        priv_mlp_units = list(self.network_config.priv_mlp.units)

        self.actor = SquashedGaussianActor(
            obs_dim, action_dim, hidden_units,
            self.priv_info_dim, priv_mlp_units
        ).to(self.device)

        self.critic = TwinQCritic(
            obs_dim, action_dim, hidden_units,
            self.priv_info_dim, priv_mlp_units
        ).to(self.device)

        self.critic_target = copy.deepcopy(self.critic)
        # Freeze target (no gradient)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ---- Automatic Entropy Tuning ----
        self.target_entropy = -action_dim  # heuristic: -dim(A)
        self.log_alpha = torch.tensor(
            np.log(self.init_temperature), dtype=torch.float32,
            device=self.device, requires_grad=True
        )

        # ---- Optimizers ----
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

        # ---- Normalization (same as PPO for compatibility) ----
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.priv_mean_std = RunningMeanStd((self.priv_info_dim,)).to(self.device)

        # ---- Replay Buffer ----
        self.replay_buffer = ReplayBuffer(
            capacity=self.buffer_size,
            obs_dim=obs_dim,
            action_dim=action_dim,
            priv_dim=self.priv_info_dim,
            num_envs=self.num_actors,
            device=self.device
        )

        # ---- Logging (same structure as PPO) ----
        if env is not None and not full_config.offline_training:
            self.output_dir = output_dif
            self.nn_dir = os.path.join(self.output_dir, 'stage1_nn')
            self.tb_dif = os.path.join(self.output_dir, 'stage1_tb')
            os.makedirs(self.nn_dir, exist_ok=True)
            os.makedirs(self.tb_dif, exist_ok=True)
            self.writer = SummaryWriter(self.tb_dif)

        # ---- Tracking ----
        self.episode_rewards = AverageScalarMeter(100)
        self.episode_lengths = AverageScalarMeter(100)
        self.episode_success = AverageScalarMeter(100)

        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config['max_agent_steps']
        self.best_rewards = -10000
        self.best_success = 0
        self.cur_reward = -10000
        self.success_rate = 0
        self.epoch_num = 0
        self.save_best_after = self.ppo_config.get('save_best_after', 1_000_000)
        self.save_freq = self.ppo_config.get('save_frequency', 100)
        self.extra_info = {}

        self.it = 0
        self.log_video_every = self.task_config.env.record_video_every
        self.last_recording_it = 0
        self.last_recording_it_ft = 0

        # Timing
        self.data_collect_time = 0
        self.rl_train_time = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ============================================================
    # Core SAC Update
    # ============================================================

    def update_critic(self, batch):
        obs = batch['obs']
        next_obs = batch['next_obs']
        actions = batch['actions']
        rewards = batch['rewards']
        dones = batch['dones']
        priv = batch['priv_info']
        next_priv = batch['next_priv_info']

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_obs, next_priv)
            target_q1, target_q2 = self.critic_target(next_obs, next_action, next_priv)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            target_value = rewards + (1.0 - dones) * self.gamma * target_q

        q1, q2 = self.critic(obs, actions, priv)
        critic_loss = F.mse_loss(q1, target_value) + F.mse_loss(q2, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        return critic_loss.item()

    def update_actor_and_alpha(self, batch):
        obs = batch['obs']
        priv = batch['priv_info']

        action, log_prob, _ = self.actor.sample(obs, priv)
        q1, q2 = self.critic(obs, action, priv)
        q = torch.min(q1, q2)

        actor_loss = (self.alpha.detach() * log_prob - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Update temperature
        alpha_loss = torch.tensor(0.0)
        if self.learnable_temperature:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        return actor_loss.item(), alpha_loss.item(), log_prob.mean().item()

    def soft_update_target(self):
        """Polyak averaging for target network."""
        for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_targ.data.mul_(1 - self.tau_soft)
            p_targ.data.add_(self.tau_soft * p.data)

    # ============================================================
    # Training Loop (mirrors PPO.train() interface)
    # ============================================================

    def set_eval(self):
        self.actor.eval()
        self.critic.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
            self.priv_mean_std.eval()

    def set_train(self):
        self.actor.train()
        self.critic.train()
        if self.normalize_input:
            self.running_mean_std.train()
            self.priv_mean_std.train()

    def train(self):
        _t = time.time()
        obs_dict = self.env.reset(reset_at_success=False, reset_at_fails=True)

        current_rewards = torch.zeros(self.num_actors, 1, device=self.device)
        current_lengths = torch.zeros(self.num_actors, device=self.device)
        current_success = torch.zeros(self.num_actors, device=self.device)

        total_updates = 0

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            _t_collect = time.time()

            # ---- Collect one step from all envs ----
            with torch.no_grad():
                obs_normalized = self.running_mean_std(obs_dict['obs'])
                priv_normalized = self.priv_mean_std(obs_dict['priv_info'])

                if self.agent_steps < self.warmup_steps:
                    # Random exploration during warmup
                    actions = 2 * torch.rand(self.num_actors, self.actions_num, device=self.device) - 1
                else:
                    actions, _, _ = self.actor.sample(obs_normalized, priv_normalized)

            actions_clamped = torch.clamp(actions, -1.0, 1.0)
            next_obs_dict, rewards, dones, infos = self.env.step(actions_clamped)

            # Store transition
            with torch.no_grad():
                next_obs_normalized = self.running_mean_std(next_obs_dict['obs'])
                next_priv_normalized = self.priv_mean_std(next_obs_dict['priv_info'])

            self.replay_buffer.add_batch(
                obs=obs_normalized,
                next_obs=next_obs_normalized,
                actions=actions_clamped,
                rewards=rewards.unsqueeze(1),
                dones=dones.float().unsqueeze(1),
                priv_info=priv_normalized,
                next_priv_info=next_priv_normalized,
            )

            # Track episodes
            current_rewards += rewards.unsqueeze(1)
            current_success += infos.get('successes', torch.zeros(self.num_actors, device=self.device))
            current_lengths += 1

            done_indices = dones.nonzero(as_tuple=False)
            self.episode_rewards.update(current_rewards[done_indices])
            self.episode_lengths.update(current_lengths[done_indices])
            self.episode_success.update(current_success[done_indices])

            not_dones = 1.0 - dones.float()
            current_rewards *= not_dones.unsqueeze(1)
            current_lengths *= not_dones
            current_success *= not_dones

            self.extra_info = {}
            for k, v in infos.items():
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.extra_info[k] = v

            obs_dict = next_obs_dict
            self.agent_steps += self.num_actors
            self.data_collect_time += (time.time() - _t_collect)

            # ---- Update networks ----
            if self.agent_steps >= self.warmup_steps and self.replay_buffer.size >= self.batch_size_sac:
                _t_train = time.time()
                self.set_train()

                for _ in range(self.updates_per_step):
                    batch = self.replay_buffer.sample(self.batch_size_sac)

                    critic_loss = self.update_critic(batch)
                    actor_loss, alpha_loss, entropy = self.update_actor_and_alpha(batch)
                    self.soft_update_target()
                    total_updates += 1

                self.rl_train_time += (time.time() - _t_train)
                self.set_eval()

                # ---- Logging ----
                if total_updates % 100 == 0 and (not self.multi_gpu or self.rank == 0):
                    mean_rewards = self.episode_rewards.get_mean()
                    mean_lengths = self.episode_lengths.get_mean()
                    mean_success = self.episode_success.get_mean()
                    self.cur_reward = mean_rewards

                    all_fps = self.agent_steps / (time.time() - _t)
                    info_string = (
                        f'Agent Steps: {int(self.agent_steps // 1e6):04}M | '
                        f'FPS: {all_fps:.1f} | '
                        f'Critic Loss: {critic_loss:.4f} | '
                        f'Actor Loss: {actor_loss:.4f} | '
                        f'Alpha: {self.alpha.item():.4f} | '
                        f'Entropy: {entropy:.3f} | '
                        f'Best Reward: {self.best_rewards:.2f} | '
                        f'Cur Reward: {mean_rewards:.2f} | '
                        f'Success: {mean_success:.2f}'
                    )
                    print(info_string)

                    self.writer.add_scalar('losses/critic_loss', critic_loss, self.agent_steps)
                    self.writer.add_scalar('losses/actor_loss', actor_loss, self.agent_steps)
                    self.writer.add_scalar('losses/alpha_loss', alpha_loss, self.agent_steps)
                    self.writer.add_scalar('info/alpha', self.alpha.item(), self.agent_steps)
                    self.writer.add_scalar('info/entropy', entropy, self.agent_steps)
                    self.writer.add_scalar('episode_rewards/step', mean_rewards, self.agent_steps)
                    self.writer.add_scalar('episode_lengths/step', mean_lengths, self.agent_steps)
                    self.writer.add_scalar('mean_success/step', mean_success, self.agent_steps)

                    for k, v in self.extra_info.items():
                        self.writer.add_scalar(f'{k}', v, self.agent_steps)

                    # Save best
                    if mean_rewards > self.best_rewards and self.agent_steps >= self.save_best_after and mean_rewards != 0:
                        print(f"save current best reward: {mean_rewards:.2f}")
                        self.best_rewards = mean_rewards
                        self.save(os.path.join(self.nn_dir, f"best_reward_{mean_rewards:.2f}"))

                    self.success_rate = mean_success

                    # ---- Curriculum auto-graduation (same logic as PPO) ----
                    current_sr = mean_success.item() if hasattr(mean_success, 'item') else mean_success
                    target_sr = 0.90
                    if current_sr >= target_sr and self.epoch_num > 5:
                        print(f"\n{'=' * 60}")
                        print(f"🎉 Success rate reached {current_sr * 100:.1f}%! Level complete!")
                        print(f"{'=' * 60}\n")
                        self.save(os.path.join(self.nn_dir, 'last'))
                        import sys
                        sys.exit(0)

            # Video logging
            if not self.multi_gpu or self.rank == 0:
                self.log_video()
                self.it += 1

        print('max steps achieved')

    # ============================================================
    # Save / Restore (compatible with your train.py)
    # ============================================================

    def save(self, name):
        weights = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'running_mean_std': self.running_mean_std.state_dict(),
            'priv_mean_std': self.priv_mean_std.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'agent_steps': self.agent_steps,
        }
        torch.save(weights, f'{name}.pth')

    def restore_train(self, fn, restore_student=False, phase=None, **kwargs):
        """
        Restore from a checkpoint. Supports loading from either:
        - A SAC checkpoint (has 'actor' key)
        - A PPO checkpoint (has 'model' key) for warm-starting from PPO
        """
        cprint(f"Restore from {fn}", 'red', attrs=['bold'])
        if not fn:
            return

        checkpoint = torch.load(fn, map_location=self.device)

        if 'actor' in checkpoint:
            # Loading from SAC checkpoint
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.log_alpha.data.copy_(checkpoint['log_alpha'].to(self.device))
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
            self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])
            if 'actor_optimizer' in checkpoint:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            if 'agent_steps' in checkpoint:
                self.agent_steps = checkpoint['agent_steps']
            cprint("Loaded SAC checkpoint successfully", 'green')

        elif 'model' in checkpoint:
            # Loading from PPO checkpoint (warm-start)
            cprint("Detected PPO checkpoint — warm-starting actor from PPO weights", 'yellow')
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
            self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])
            # NOTE: PPO model architecture differs, so we only load normalization stats.
            # Actor/critic will train from scratch but benefit from normalized inputs.
            cprint("Loaded normalization stats from PPO. Actor/Critic training from scratch.", 'yellow')

    def restore_test(self, fn):
        checkpoint = torch.load(fn, map_location=self.device)
        if 'actor' in checkpoint:
            self.actor.load_state_dict(checkpoint['actor'])
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
            self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])
        elif 'model' in checkpoint:
            cprint("Warning: Loading PPO checkpoint for SAC test — only normalization loaded", 'yellow')
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
            self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])

    def test(self, milestone=100, total_steps=1e9):
        self.set_eval()
        steps = 0
        total_dones, num_success = 0, 0
        obs_dict = self.env.reset(reset_at_success=False, reset_at_fails=False)

        while steps < total_steps:
            steps += 1
            with torch.no_grad():
                obs_norm = self.running_mean_std(obs_dict['obs'])
                priv_norm = self.priv_mean_std(obs_dict['priv_info'])
                action = self.actor.act_deterministic(obs_norm, priv_norm)
                action = torch.clamp(action, -1.0, 1.0)

            obs_dict, r, done, info = self.env.step(action)

            if self.env.progress_buf[0] == self.env.max_episode_length - 1:
                num_success = self.env.success_reset_buf[done.nonzero()].sum()
                total_dones = len(done.nonzero())
                success_rate = num_success / total_dones if total_dones > 0 else 0
                print(f'[Test] success rate: {success_rate:.3f}')
                break

        if total_dones > 0 and success_rate > self.best_success and self.agent_steps > 1e5:
            self.best_success = success_rate
            self.save(os.path.join(self.nn_dir, f'best_succ_{success_rate:.2f}'))

        return num_success, total_dones

    # ============================================================
    # Video logging (same as PPO)
    # ============================================================

    def log_video(self):
        if self.it == 0:
            self.env.start_recording()
            self.last_recording_it = self.it
            self.env.start_recording_ft()
            self.last_recording_it_ft = self.it
            return

        frames = self.env.get_complete_frames()
        ft_frames = self.env.get_ft_frames()
        if len(frames) > 0:
            self.env.pause_recording()
            self.env.pause_recording_ft()

            if len(frames) < 20:
                self.env.start_recording()
                self.last_recording_it = self.it
                self.env.start_recording_ft()
                self.last_recording_it_ft = self.it
                return

            video_dir = os.path.join(self.output_dir, 'videos1')
            os.makedirs(video_dir, exist_ok=True)

            writer = imageio.get_writer(f"{video_dir}/{self.it:05d}.mp4", mode='I', fps=30)
            for i in range(len(frames)):
                writer.append_data(np.uint8(frames[i]))
            writer.close()

            self.env.start_recording()
            self.last_recording_it = self.it
            self.env.start_recording_ft()
            self.last_recording_it_ft = self.it