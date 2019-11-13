#!/usr/bin/env python
# -*- coding: utf-8 -*-


import random
import os
import torch
import numpy as np
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from unityagents import UnityEnvironment
from agent import Agent


def set_seed(seed):
    """Set the seed for pseudo-random number generations

    Args:
        seed (int): seed to set for reproducibility
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_agent(agent, env, n_episodes=7500, max_t=1000, success_thresh=0.5):
    """Agent training function

    Args:
        agent: agent to train
        env: environment to interact with
        n_episodes (int, optional): maximum number of training episodes
        max_t (int, optional): maximum number of timesteps per episode
        success_thresh (float, optional): minimum running average score to consider environment solved

    Returns:
        scores (list<float>): scores of each episode
    """
    scores = []
    # Last 100 episodes' scores
    scores_window = deque(maxlen=100)
    brain_name = env.brain_names[0]
    success = False
    tqdm_range = trange(1, n_episodes + 1)
    for i_episode in tqdm_range:
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        # get the current state
        states = env_info.vector_observations
        agent.reset()
        # initialize the score
        _scores = np.zeros(len(env_info.agents))
        for t in range(max_t):
            actions = agent.act(states, add_noise=True)
            # Perform action in the environment
            env_info = env.step(actions)[brain_name]
            # Get next state, reward and completion boolean
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            # Agent step
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done, t)
            # Update episode score
            states = next_states
            _scores += rewards
            if np.any(dones):
                break
        # Save most recent score
        scores.append(np.mean(_scores))
        scores_window.append(scores[-1])

        # Console output
        running_mean = np.mean(scores_window)
        tqdm_range.set_postfix(raw_score=scores[-1], avg_score=running_mean)
        if i_episode % 100 == 0:
            tqdm.write(f'Episode {i_episode}/{n_episodes}\tavg score: {running_mean:.2f}')
        if (not success) and running_mean >= success_thresh:
            tqdm.write(f'Solved in {i_episode:d} episodes!\tavg score: {running_mean:.2f}')
            success = True

    return scores


def plot_scores(scores, running_window_size=100, success_thresh=30.):
    """Plot the score statistics over training episodes

    Args:
        scores (list<float>): scores obtained at each episode
        running_window_size (int, optional): number of episodes used to moving window
        success_thresh (float): minimum score to consider the environment as solved
    """

    # plot the scores
    plt.plot(np.arange(len(scores)), scores, zorder=1)
    # Running average scores
    ra_scores, rm_scores = [], []
    success_x, success_y = None, None
    success = False
    for idx in range(len(scores)):
        ra_score = np.mean(scores[max(0, idx - running_window_size + 1): idx + 1])
        ra_scores.append(ra_score)
        rm_scores.append(np.median(scores[max(0, idx - running_window_size + 1): idx + 1]))
        if (not success) and ra_score > success_thresh:
            success_x, success_y = idx + 1, ra_score
            success = True
    plt.plot(np.arange(len(scores)), ra_scores, zorder=2)
    plt.plot(np.arange(len(scores)), rm_scores, zorder=3)
    plt.axhline(y=success_thresh, color='r', linestyle='--', zorder=4)
    if success_x and success_y:
        plt.scatter(success_x, success_y, color='r', zorder=5)
    # Legend
    plt.grid(True, linestyle='dotted')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    legends = ['Raw score', 'Running average score', 'Running median score']
    if success_x and success_y:
        legends.append('Success episode')
    plt.legend(legends, loc='upper right')
    plt.title('DDPG training scores')
    plt.show()


def main(args):

    if args.deterministic:
        set_seed(42)

    env = UnityEnvironment(file_name=args.env_path, no_graphics=args.no_graphics)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    agent = Agent(state_size, action_size,
                  train=True,
                  device=args.device,
                  buffer_size=args.buffer_size,
                  batch_size=args.batch_size,
                  lr=args.lr,
                  gamma=args.gamma, tau=args.tau,
                  update_freq=args.update_freq, nb_updates=args.nb_updates,
                  noise_mean=args.noise_mean, noise_theta=args.noise_theta,
                  noise_sigma=args.noise_sigma, eps=args.eps, eps_decay=args.eps_decay,
                  grad_clip=args.grad_clip)

    scores = train_agent(agent, env,
                         n_episodes=args.episodes)

    output_folder = Path(args.output)
    if not output_folder.is_dir():
        output_folder.mkdir(parents=True)
    # Save model
    torch.save(agent.actor_local.state_dict(), output_folder.joinpath('actor_model.pt'))
    torch.save(agent.critic_local.state_dict(), output_folder.joinpath('critic_model.pt'))

    env.close()

    # Plot results
    fig = plt.figure()
    plot_scores(scores, running_window_size=100, success_thresh=args.success_threshold)
    fig.savefig(output_folder.joinpath('training_scores.png'), transparent=True)


if __name__ == "__main__":
    import argparse
    # Environment
    parser = argparse.ArgumentParser(description='Tennis agent training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--no-graphics", dest="no_graphics",
                        help="Should graphical environment be disabled",
                        action="store_true")
    # Input / Output
    parser.add_argument('--env-path', default='./Tennis_Linux/Tennis.x86_64',
                        help='path to executable unity environment')
    parser.add_argument('--output', default='./outputs', type=str, help='output folder')
    parser.add_argument('--success-threshold', default=0.5, type=float,
                        help='minimum running average score over last 100 episodes to consider environment solved')
    # Device
    parser.add_argument('--device', default=None, help='device')
    parser.add_argument("--deterministic", dest="deterministic",
                        help="should the training be performed in deterministic mode",
                        action="store_true")
    # Loader
    parser.add_argument('-b', '--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--buffer-size', default=1e6, type=int, help='replay buffer size')
    # Optimizer
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--episodes', default=7500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--gamma', default=0.99, type=float, help='discount factor')
    parser.add_argument('--tau', default=1e-3, type=float, help='for soft update of target parameters')
    parser.add_argument('--update-freq', default=20, type=int, help='number of steps between each updates')
    parser.add_argument('--nb-updates', default=10, type=int, help='number of updates to perform')
    parser.add_argument('--noise-mean', default=0., type=float, help='mean of Ornstein-Uhlenbeck noise')
    parser.add_argument('--noise-theta', default=0.05, type=float, help='theta parameter Ornstein-Uhlenbeck noise')
    parser.add_argument('--noise-sigma', default=0.15, type=float, help='sigma parameter of Ornstein-Uhlenbeck noise')
    parser.add_argument('--eps', default=1.0, type=float, help='initial noise scaling')
    parser.add_argument('--eps-decay', default=1e-6, type=float, help='noise scaling decay')
    parser.add_argument('--grad-clip', default=1., type=float, help='gradient clipping')
    args = parser.parse_args()

    main(args)
