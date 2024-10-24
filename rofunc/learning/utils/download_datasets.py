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

import collections
import os
import pickle

import gym
import numpy as np

import rofunc as rf
from rofunc.utils.logger.beauty_logger import beauty_print


def download_d4rl_dataset(save_dir):
    rf.oslab.create_dir(save_dir)

    for env_name in ['halfcheetah', 'hopper', 'walker2d']:
        for dataset_type in ['medium', 'medium-replay', 'expert']:
            name = f'{env_name}-{dataset_type}-v2'

            if os.path.exists(f'{save_dir}/{name}.pkl'):
                continue

            import d4rl  # Import required to register environments, you may need to also import the submodule

            env = gym.make(name)
            dataset = env.get_dataset()

            N = dataset['rewards'].shape[0]
            data_ = collections.defaultdict(list)

            use_timeouts = False
            if 'timeouts' in dataset:
                use_timeouts = True

            episode_step = 0
            paths = []
            for i in range(N):
                done_bool = bool(dataset['terminals'][i])
                if use_timeouts:
                    final_timestep = dataset['timeouts'][i]
                else:
                    final_timestep = (episode_step == 1000 - 1)
                for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
                    data_[k].append(dataset[k][i])
                if done_bool or final_timestep:
                    episode_step = 0
                    episode_data = {}
                    for k in data_:
                        episode_data[k] = np.array(data_[k])
                    paths.append(episode_data)
                    data_ = collections.defaultdict(list)
                episode_step += 1

            returns = np.array([np.sum(p['rewards']) for p in paths])
            num_samples = np.sum([p['rewards'].shape[0] for p in paths])
            print(f'Number of samples collected: {num_samples}')
            print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)},'
                  f' min = {np.min(returns)}')

            with open(f'{save_dir}/{name}.pkl', 'wb') as f:
                pickle.dump(paths, f)

    beauty_print('D4RL dataset downloaded', type='info')

# if __name__ == '__main__':
#     download_d4rl_dataset()
