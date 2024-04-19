from src.environments.env_bit_sequence_flipping_rng_target import FlippingBitSequenceEnvRNGTarget
from src.agents.agent_dqn_target import DQNAgentTarget
from src.agents.agent_dqn_target_her import DQNAgentTargetHER
from src.models.model_dqn_bitflipping_target import BitFlippingDQNNetworkTarget, BitFlippingDQNNetworkTargetUVFA, BitFlippingDQNNetworkTargetHandCrafted
from src.models.buffer_bitflipping_target import BufferBitflippingTarget

import numpy as np
import torch

from tqdm import tqdm
import matplotlib.pyplot as plt

NORMAL = "normal"
UVFA = "uvfa"
HANDCRAFTED = "handcrafted"

def get_new_model(model_type: str, n: int):
    assert model_type in [NORMAL, UVFA, HANDCRAFTED], "Model type must be one of 'normal', 'uvfa', 'handcrafted'"
    if model_type == NORMAL:
        return BitFlippingDQNNetworkTarget(n)
    elif model_type == UVFA:
        return BitFlippingDQNNetworkTargetUVFA(n)
    elif model_type == HANDCRAFTED:
        return BitFlippingDQNNetworkTargetHandCrafted(n)

def train_DQN_agent(n: int, device: torch.device,
    episodes: int=10000, num_explore_agents: int=16, num_valid_agents: int=1024,
    batch_size: int=2048,
    use_HER: bool=False,
    model_type: str="uvfa") -> None:
    assert model_type in [NORMAL, UVFA, HANDCRAFTED], "Model type must be one of 'normal', 'uvfa', 'handcrafted'"

    # initialize environment, model and agent
    env = FlippingBitSequenceEnvRNGTarget(n, device)
    model = get_new_model(model_type, n)
    model.to(device)
    running_model = get_new_model(model_type, n)
    running_model.to(device)
    buffer = BufferBitflippingTarget(n, device=device, max_buffer_size=2 ** 15)
    if use_HER:
        agent = DQNAgentTargetHER(model=model, running_model=running_model,
                        buffer=buffer, device=device,
                        action_space_size=n)
    else:
        agent = DQNAgentTarget(model=model, running_model=running_model,
                        buffer=buffer, device=device,
                        action_space_size=n)
    
    target_value = n / 2

    success_rates = []
    steps_to_success = []
    loss_values = []
    mean_distances = []

    # initialize trajectory tensors
    num_steps = n # max number of steps is n
    traj_states = torch.zeros((num_steps, num_explore_agents, n), dtype=torch.int64, device=device)
    traj_actions = torch.zeros((num_steps, num_explore_agents), dtype=torch.int64, device=device)
    traj_rewards = torch.zeros((num_steps, num_explore_agents), dtype=torch.float32, device=device)
    traj_next_states = torch.zeros((num_steps, num_explore_agents, n), dtype=torch.int64, device=device)
    traj_dones = torch.zeros((num_steps, num_explore_agents), dtype=torch.bool, device=device)
    traj_goals = torch.zeros((num_steps, num_explore_agents, n), dtype=torch.int64, device=device)
    traj_step_valid = torch.zeros((num_steps, num_explore_agents), dtype=torch.bool, device=device)

    for e in tqdm(range(episodes)):
        # initial state is numpy array, convert to torch tensor
        state, target = env.reset(num_agents=num_explore_agents)
        # reset trajectory step valid
        traj_step_valid.fill_(False)

        for step in range(num_steps):  # max time steps
            with torch.no_grad():
                action = agent.act(state, target)
                next_state, reward, done, info = env.step(action)
                prev_done = info["previous_done"] # tensor indicating if the prior state was already terminal
                prev_step_ongoing = ~prev_done

                # record in trajectory
                traj_states[step, ...].copy_(state)
                traj_actions[step, ...].copy_(action)
                traj_rewards[step, ...].copy_(reward)
                traj_next_states[step, ...].copy_(next_state)
                traj_dones[step, ...].copy_(done)
                traj_goals[step, ...].copy_(target)
                traj_step_valid[step, ...].copy_(prev_step_ongoing)

                state = next_state
                if done.all():
                    break
        
        # let agent remember the trajectory
        with torch.no_grad():
            agent.remember_trajectory(traj_states, traj_actions, traj_rewards, traj_next_states, traj_dones, traj_goals, traj_step_valid)

        # replay to update Q network
        loss = agent.replay(batch_size=batch_size)

        # now run the agent to see if it has learned (without random exploration, and without affecting the replay buffer)
        # we use random initial states
        state, target = env.reset(num_agents=num_valid_agents)
        agent_steps_to_success = np.ones(num_valid_agents)
        for step in range(num_steps):  # max time steps
            with torch.no_grad():
                action = agent.act(state, target, explore=False)
                next_state, reward, done, info = env.step(action)
                prev_done = info["previous_done"] # tensor indicating if the prior state was already terminal
                prev_step_ongoing = ~prev_done
                state = next_state

                if step == 0: # if by random chance the agent is already done in the first step, we need to handle this case
                    agent_steps_to_success[prev_done.cpu().numpy()] = 0
                agent_steps_to_success[~done.cpu().numpy()] += 1
                if done.all():
                    break
        success_rate = ((done.sum().item() + 0.0) / num_valid_agents)
        avg_steps_to_success = agent_steps_to_success.mean()
        with torch.no_grad():
            mean_distance = env.distance_to_target(state).to(torch.float32).mean().item()
        success_rates.append(success_rate)
        steps_to_success.append(avg_steps_to_success)
        loss_values.append(loss)
        mean_distances.append(mean_distance)

        if e > 100 and (np.allclose(np.array(success_rates[-100:]), 1.0)):
            print("Early stopping at episode", e)
            print("Episode: {}/{}, Epsilon: {}, Success rate: {:.2f}, Avg steps to success: {:.2f}, Loss: {:.6f}".format(e, episodes, agent.epsilon, success_rate, avg_steps_to_success, loss))
            break

        if (e + 1) % 100 == 0:
            print("Episode: {}/{}, Epsilon: {}, Success rate: {:.2f}, Avg steps to success: {:.2f}, Loss: {:.6f}".format(e, episodes, agent.epsilon, success_rate, avg_steps_to_success, loss))
    
    # plot success rates and steps to success over episodes
    plt.figure(figsize=(24, 16))
    plt.subplot(2, 3, 1)
    plt.plot(success_rates)
    plt.xlabel("Episodes")
    plt.ylabel("Success rate")
    plt.title("Success rate over episodes")
    plt.subplot(2, 3, 2)
    plt.plot(steps_to_success)
    plt.xlabel("Episodes")
    plt.ylabel("Steps to success")
    plt.title("Steps to success over episodes")
    plt.subplot(2, 3, 3)
    plt.plot(steps_to_success)
    plt.ylim(target_value - 1, target_value + 1)
    plt.xlabel("Episodes")
    plt.ylabel("Steps to success")
    plt.title("Steps to success over episodes (Zoom)")
    plt.subplot(2, 3, 4)
    plt.plot(loss_values)
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.title("Loss over episodes")
    plt.subplot(2, 3, 5)
    plt.plot(mean_distances)
    plt.xlabel("Mean distance to target")
    plt.ylabel("Loss")
    plt.title("Mean distance to target over episodes")
    plt.show()

    return agent, env