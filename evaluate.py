#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import torch
from pathlib import Path
from unityagents import UnityEnvironment
from agent import Agent


def evaluate_agent(agent, env, runs=10):
    """Agent training function

    Args:
        agent: agent to train
        env: environment to interact with

    Returns:
        score (float): total score of episode
    """
    brain_name = env.brain_names[0]
    scores = np.zeros(runs)
    for idx in range(runs):
        # reset the environment
        env_info = env.reset(train_mode=False)[brain_name]
        # get the current state
        states = env_info.vector_observations
        # initialize the score
        _scores = np.zeros(len(env_info.agents))
        while True:
            actions = agent.act(states, add_noise=False)
            # Perform action in the environment
            env_info = env.step(actions)[brain_name]
            # Get next state, reward and completion boolean
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            # Update episode score
            states = next_states
            _scores += rewards
            if np.any(dones):
                break
        scores[idx] = np.mean(_scores)
    return np.mean(scores)


def main(args):

    env = UnityEnvironment(file_name=args.env_path)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    state_size = len(state)
    print('Number of states:', state_size)

    # Instantiate trained agent
    agent = Agent(state_size, action_size,
                  train=False,
                  device=args.device)

    # Load checkpoint
    checkpoint = Path(args.checkpoint)
    if checkpoint.is_file():
        state_dict = torch.load(checkpoint, map_location=args.device)
    else:
        raise FileNotFoundError(f"unable to locate {args.checkpoint}")
    agent.actor_local.load_state_dict(state_dict)

    # Evaluation run
    score = evaluate_agent(agent, env, runs=args.runs)

    print(f"Evaluation score: {score}")
    env.close()


if __name__ == "__main__":
    import argparse
    # Environment
    parser = argparse.ArgumentParser(description='Tennis player evaluator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--no-graphics", dest="no_graphics",
                        help="Should graphical environment be disabled",
                        action="store_true")
    parser.add_argument("--runs", default=10, type=int, help='Number of evaluation runs to perform')
    # Input / Output
    parser.add_argument('--env-path', default='./Tennis_Linux/Tennis.x86_64',
                        help='path to executable unity environment')
    parser.add_argument('--checkpoint', default='./outputs/model.pt', type=str,
                        help='model state dict to load')
    # Device
    parser.add_argument('--device', default=None, help='device')
    args = parser.parse_args()

    main(args)
