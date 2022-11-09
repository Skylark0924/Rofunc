import torchvision
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.functional as F
from matplotlib import pyplot as plt
from matplotlib import style


def data_process(states, actions=None, n=2, compare=1):
    count = 0
    index = []

    if type(actions) != torch.Tensor:
        ep, t, state_size = states.shape
    else:
        ep, t, state_size = states.shape
        _, _, action_size = actions.shape

    if type(actions) != torch.Tensor:
        output_states = torch.zeros((ep * (t - n + 1), state_size * n), dtype=torch.float)
    else:
        output_states = torch.zeros((ep * (t - n + 1), state_size * n), dtype=torch.float)
        output_actions = torch.zeros((ep * (t - n + 1), action_size), dtype=torch.float)

    for i in range(ep):
        for j in range(t - n + 1):
            if (states[i, j] == -compare * torch.ones(state_size)).all() or (
                    states[i, j + 1] == -compare * torch.ones(state_size)).all():
                index.append([i, j])
            else:
                output_states[count] = states[i, j:j + n].view(-1)

            if type(actions) != torch.Tensor:
                count += 1
                # do nothing
            else:
                output_actions[count] = actions[i, j]
                count += 1

    if type(actions) != torch.Tensor:
        output_states = output_states[:count]
        return output_states
    else:
        output_states = output_states[:count]
        output_actions = output_actions[:count]
        return output_states, output_actions


def train_transition (training_set, state_space_size, model, n=2,   batch_size = 256, n_epoch = 50):
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_list = []
    for itr in range(n_epoch):
        total_loss = 0
        b=0
        for batch in range (0,training_set.shape[0], batch_size):
            data   = training_set  [batch : batch+batch_size , :n*state_space_size]
            y      = training_set [batch : batch+batch_size, n*state_space_size:]
            y_pred = model(data)
            loss   = criterion(y_pred, y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            b += 1
        print("[EPOCH]: %i, [LOSS]: %.6f" % (itr+1, total_loss/b))
        loss_list.append(total_loss / training_set.shape[0])
    return model


def train_policy(training_set, state_space_size, policy, batch_size=256, n_epoch=50):
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    loss_list = []
    for itr in range(n_epoch):
        total_loss = 0
        b = 0
        for batch in range(0, training_set.shape[0], batch_size):
            data = training_set[batch: batch + batch_size, :state_space_size]
            y = training_set[batch: batch + batch_size, state_space_size:]
            y_pred = policy(data)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            b += 1
        print("[EPOCH]: %i, [LOSS]: %.6f" % (itr + 1, total_loss / b))
        loss_list.append(total_loss / training_set.shape[0])
    return policy

# behavior cloning
if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    state_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.shape[0]
    expert_states = torch.tensor(np.load('../../../data/BCO/states_expert_Pendulum.npy'), dtype=torch.float)
    transition_model = nn.Sequential(
        nn.Linear(state_space_size * 2, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, state_space_size)
    )
    policy_model = nn.Sequential(
        nn.Linear(state_space_size, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, action_space_size)
    )
    ep = 100
    t = 100
    states = torch.zeros((ep, t, state_space_size), dtype=torch.float)
    actions = torch.zeros((ep, t, action_space_size), dtype=torch.float)
    for i in range(ep):
        state = env.reset()
        for j in range(t):
            states[i, j] = torch.tensor(state, dtype=torch.float)
            action = env.action_space.sample()
            actions[i, j] = torch.tensor(action, dtype=torch.float)
            state, reward, done, info = env.step(action)
            if done:
                break
    states, actions = data_process(states, actions)
    training_set = torch.cat((states, actions), dim=1)
    transition_model = train_transition(training_set, state_space_size, transition_model)
    policy_model = train_policy(training_set, state_space_size, policy_model)
    torch.save(transition_model, 'transition_model.pt')
    torch.save(policy_model, 'policy_model.pt')
