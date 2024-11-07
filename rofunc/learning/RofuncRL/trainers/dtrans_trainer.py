#  Copyright (C) 2024, Junjia Liu
#
#  This file is part of Rofunc.
#
#  Rofunc is licensed under the GNU General Public License v3.0.
#  You may use, distribute, and modify this code under the terms of the GPL-3.0.
#
#  Additional Terms for Commercial Use:
#  Commercial use requires sharing 50% of net profits with the copyright holder.
#  Financial reports and regular payments must be provided as agreed in writing.
#  Non-compliance results in revocation of commercial rights.
#
#  For more details, see <https://www.gnu.org/licenses/>.
#  Contact: skylark0924@gmail.com
import copy
import os
import pickle
import random

import numpy as np
import torch
import tqdm

from rofunc.learning.RofuncRL.agents.offline.dtrans_agent import DTransAgent
from rofunc.learning.RofuncRL.trainers.base_trainer import BaseTrainer


def discount_cumsum(x, gamma):
    tmp = np.zeros_like(x)
    tmp[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        tmp[t] = x[t] + gamma * tmp[t + 1]
    return tmp


class DTransTrainer(BaseTrainer):
    def __init__(self, cfg, env, device, env_name, **kwargs):
        super().__init__(cfg, env, device, env_name, **kwargs)
        self.agent = DTransAgent(cfg.train, self.env.observation_space, self.env.action_space, device, self.exp_dir,
                                 self.rofunc_logger)

        self.pct_traj = 1
        self.dataset_type = self.cfg.train.Trainer.dataset_type
        self.dataset_root_path = self.cfg.train.Trainer.dataset_root_path
        self.mode = self.cfg.train.Trainer.mode
        self.scale = self.cfg.train.Trainer.scale
        self.max_episode_steps = self.cfg.train.Trainer.max_episode_steps
        self.max_seq_length = self.cfg.train.Trainer.max_seq_length

        self.loss_mean = 0

        # list of dict, each dict contains a traj with
        # ['observations', 'next_observations', 'actions', 'rewards', 'terminals']
        self.trajectories = None

        self.load_dataset()

    def load_dataset(self):
        """
        Load dataset from pickle file and preprocess it.
        """
        dataset_path = os.path.join(self.dataset_root_path, f'{self.env_name.lower()}-{self.dataset_type}-v2.pkl')
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        # save all path information into separate lists
        states, traj_lens, returns = [], [], []
        for path in self.trajectories:
            if self.mode == 'delayed':  # delayed: all rewards moved to end of trajectory
                path['rewards'][-1] = path['rewards'].sum()
                path['rewards'][:-1] = 0.
            states.append(path['observations'])
            traj_lens.append(len(path['observations']))
            returns.append(path['rewards'].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        num_timesteps = sum(traj_lens)
        num_timesteps = max(int(self.pct_traj * num_timesteps), 1)
        sorted_inds = np.argsort(returns)  # lowest to highest

        self.num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(self.trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            self.num_trajectories += 1
            ind -= 1
        self.sorted_inds = sorted_inds[-self.num_trajectories:]

        # used to re-weight sampling, so we sample according to timesteps instead of trajectories
        self.p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

        self.rofunc_logger.module(f'Starting new experiment: {self.env_name} {self.dataset_type}'
                                  f' with {len(traj_lens)} trajectories and {num_timesteps} timesteps'
                                  f' Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}'
                                  f' Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')

    def get_batch(self, batch_size=256):
        state_dim = self.agent.dtrans.state_dim
        act_dim = self.agent.dtrans.action_dim

        batch_inds = np.random.choice(
            np.arange(self.num_trajectories),
            size=batch_size,
            replace=True,
            p=self.p_sample,  # re-weights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = self.trajectories[int(self.sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + self.max_seq_length].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + self.max_seq_length].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + self.max_seq_length].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + self.max_seq_length].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + self.max_seq_length].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_episode_steps] = self.max_episode_steps - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, self.max_seq_length - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - self.state_mean) / self.state_std
            a[-1] = np.concatenate([np.ones((1, self.max_seq_length - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, self.max_seq_length - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, self.max_seq_length - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, self.max_seq_length - tlen, 1)), rtg[-1]], axis=1) / self.scale
            timesteps[-1] = np.concatenate([np.zeros((1, self.max_seq_length - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, self.max_seq_length - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=self.device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=self.device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=self.device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=self.device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=self.device)

        return s, a, r, d, rtg, timesteps, mask

    def train(self):
        """
        Main training loop.
        """
        with tqdm.trange(self.maximum_steps, ncols=80, colour='green') as self.t_bar:
            for _ in self.t_bar:
                batch = self.get_batch()
                self.agent.update_net(batch)
                self.post_interaction()
                self._step += 1

        # close the logger
        self.writer.close()
        self.rofunc_logger.info('Training complete.')

    def post_interaction(self):
        # Update best models and tensorboard
        if not self._step % self.write_interval and self.write_interval > 0:
            # update best models
            self.loss_mean = np.mean(self.agent.tracking_data.get("Loss", -1e4))
            if self.loss_mean < self.agent.checkpoint_best_modules["loss"]:
                self.agent.checkpoint_best_modules["timestep"] = self._step
                self.agent.checkpoint_best_modules["loss"] = self.loss_mean
                self.agent.checkpoint_best_modules["saved"] = False
                self.agent.checkpoint_best_modules["modules"] = {k: copy.deepcopy(self.agent._get_internal_value(v)) for
                                                                 k, v in self.agent.checkpoint_modules.items()}
                self.agent.save_ckpt(os.path.join(self.agent.checkpoint_dir, "best_ckpt.pth"))

            # Update tensorboard
            self.write_tensorboard()

            # Update tqdm bar message
            if self.eval_flag:
                post_str = f"Loss/Best/Eval: {self.loss_mean:.2f}/{self.agent.checkpoint_best_modules['loss']:.2f}/{self.eval_loss_mean:.2f}"
            else:
                post_str = f"Loss/Best: {self.loss_mean:.2f}/{self.agent.checkpoint_best_modules['loss']:.2f}"
            self.t_bar.set_postfix_str(post_str)
            self.rofunc_logger.info(f"Step: {self._step}, {post_str}", local_verbose=False)

        # Save checkpoints
        if self.agent.checkpoint_interval is not None:
            if not (self._step + 1) % self.agent.checkpoint_interval and \
                    self.agent.checkpoint_interval > 0 and self._step > 1:
                self.agent.save_ckpt(os.path.join(self.agent.checkpoint_dir, f"ckpt_{self._step + 1}.pth"))
