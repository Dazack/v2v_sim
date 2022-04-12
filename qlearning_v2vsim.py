#########
# Description: This code is used for applying Q-Learning to a VANET Gym enviorment.
#
# Helper Docs:
#
# https://pythonprogramming.net/q-learning-analysis-reinforcement-learning-python-tutorial/
# https://medium.com/swlh/using-q-learning-for-openais-cartpole-v1-4a216ef237df
#########
# Author: Phillip Garrad


import gym # openAi gym
import numpy as np
import time
import math
import json
import csv
import logging.config
from v2v_sim.envs.V2VSimulation import V2VSimulationEnv



## Load a scenario Configuration File
scenario = "scenario2"
with open(f'./scenarios/{scenario}.json') as json_file:
    data = json.load(json_file)

## Load the V2V simulation Enviroment
env = gym.make("V2VSimulation-v0")
env = V2VSimulationEnv(data)
env.reset()
# env.render()

# Turning off logging ptherwise get ~Gbs worth of logs very quickly

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True
})

### Q Learning Setup inputs for Bellman Equaltion
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 80000
total_reward = 0
prior_reward = 0
step_cnt_total = 0
DISCRETE_OS_SIZE = [2] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
epsilon = 1
epsilon_decay_value = 0.99995
summary = []

num_box = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
q_table = np.zeros(num_box + (env.action_space.n,))

q_table.shape

# Prepare csv writter
csv_file = open(f'qlearning_results/{scenario}__0_1__0_95__{EPISODES}.csv', 'w', encoding='UTF8', newline='')
writer = csv.writer(csv_file)
header = ["Episode", "Average Number of steps", "Mean Reward"]
writer.writerow(header)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table


for episode in range(EPISODES + 1): #Run all episodes
    discrete_state = get_discrete_state(env.reset()) # Get the current State of env, (Reset state)
    done = False
    episode_reward = 0 # Reset the episodes reward
    step_cnt = 0 # Reset the episodes step count

    if episode % 2000 == 0: # Print Step Number for every 2000 Episodes
        print("Episode: " + str(episode))

    # Start Simulation loop, when simulation is complete it will back out
    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state]) # Take action assigned in Q-Table
        else:
            action = np.random.randint(0, env.action_space.n) # Take a random action

        new_state, reward, done, _ = env.step(action) # Run Action in enviroment

        episode_reward += reward # Append the step reward

        new_discrete_state = get_discrete_state(new_state) #Update new state

        if episode % 2000 == 0: # Render every 2000 Episodes
            env.render()

        if not done: # Update Q-Table based on new value
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            # Run Bellman Equation to get the new Q value and update table
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        discrete_state = new_discrete_state
        step_cnt += 1 # Increment Step count
    # End of Simulation Loop

    if epsilon > 0.05: # Epsilon modification - Applying the decay Rate
        if episode_reward > prior_reward and episode > 10000:
            epsilon = math.pow(epsilon_decay_value, episode - 10000)

            if episode % 500 == 0:
                print("Epsilon: " + str(epsilon))

    step_cnt_total += step_cnt # Step total count

    total_reward += episode_reward # Episode total reward
    prior_reward = episode_reward

    if episode % 1000 == 0: # Every 1000 episodes print the average step count and the average reward
        summary.append(f"Episode: {episode}")

        # Print Average step count from past 1000 episodes
        mean_step = step_cnt_total / 1000
        stepavg = f"Average Number of steps: {round(mean_step, 3)}"
        summary.append(stepavg)

        # Print Average reward count from past 1000 episodes
        mean_reward = total_reward / 1000
        mean_rwd = f"Mean Reward: {round(mean_reward, 3)}"
        summary.append(mean_rwd)

        # Reset totals for next 1000 epsiodes
        total_reward = 0
        step_cnt_total = 0

        # Print current output of Q-Learning
        print("\n".join(summary))
        data = [episode, mean_step, mean_reward]
        writer.writerow(data)

#Close simulation and print final output of Q-Learning
csv_file.close()
env.close()
print(q_table)
print("\n".join(summary))