import gym # openAi gym
from gym import envs
import numpy as np
import datetime
import time # to get the time
import math # needed for calculations
import keras
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from time import sleep
from v2v_sim.envs.V2VSimulation import V2VSimulationEnv

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
print("OK")

## Load a scenario Configuration File
with open('./scenarios/scenario2.json') as json_file:
    data = json.load(json_file)

# env = gym.make('Taxi-v3')
env = gym.make("V2VSimulation-v0")
env = V2VSimulationEnv(data)
env.reset()
env.render()

## Let's first do some random steps in the simulation so you see how the simulation looks like

# rew_tot=0
# obs= env.reset()
# env.render()
# for _ in range(30):
#     action = env.action_space.sample() #take step using random action from possible actions (actio_space)
#     print(action)
#     obs, rew, done, info = env.step(action)
#     rew_tot = rew_tot + rew
#     env.render()
# #Print the reward of these random action
# print("Reward: %r" % rew_tot)


### Q Learning
## Helpers
# https://pythonprogramming.net/q-learning-analysis-reinforcement-learning-python-tutorial/
# https://medium.com/swlh/using-q-learning-for-openais-cartpole-v1-4a216ef237df

LEARNING_RATE = 0.1

DISCOUNT = 0.95
EPISODES = 60000
total = 0
total_reward = 0
prior_reward = 0
Observation = [42, 42, 2, 2]
np_array_win_size = np.array([1, 1, 0.01, 0.01])
DISCRETE_OS_SIZE = [2] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
epsilon = 1
epsilon_decay_value = 0.99995
summary = []

num_box = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
q_table = np.zeros(num_box + (env.action_space.n,))

# q_table = np.random.uniform(low=-20, high=20, size=(Observation + [env.action_space.n]))
q_table.shape

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table


for episode in range(EPISODES + 1): #go through the episodes
    t0 = time.time() #set the initial time
    discrete_state = get_discrete_state(env.reset()) #get the discrete start for the restarted environment 
    done = False
    episode_reward = 0 #reward starts as 0 for each episode

    if episode % 2000 == 0: 
        print("Episode: " + str(episode))

    while not done: 

        if np.random.random() > epsilon:

            action = np.argmax(q_table[discrete_state]) #take cordinated action
        else:

            action = np.random.randint(0, env.action_space.n) #do a random ation

        new_state, reward, done, _ = env.step(action) #step action to get new states, reward, and the "done" status.

        episode_reward += reward #add the reward

        new_discrete_state = get_discrete_state(new_state)

        if episode % 2000 == 0: #render
            env.render()

        # print(new_discrete_state)

        if not done: #update q-table
            max_future_q = np.max(q_table[new_discrete_state])

            current_q = q_table[discrete_state + (action,)]

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            q_table[discrete_state + (action,)] = new_q

        discrete_state = new_discrete_state

    if epsilon > 0.05: #epsilon modification
        if episode_reward > prior_reward and episode > 10000:
            epsilon = math.pow(epsilon_decay_value, episode - 10000)

            if episode % 500 == 0:
                print("Epsilon: " + str(epsilon))

    t1 = time.time() #episode has finished
    episode_total = t1 - t0 #episode total time
    total = total + episode_total

    total_reward += episode_reward #episode total reward
    prior_reward = episode_reward

    if episode % 1000 == 0: #every 1000 episodes print the average time and the average reward
        summary.append(f"Episode: {episode}")
        mean = total / 1000
        timeavg = f"Time Average: {mean}"
        summary.append(timeavg)
        total = 0

        mean_reward = total_reward / 1000
        mean_rwd = f"Mean Reward: {mean_reward}"
        summary.append(mean_rwd)
        total_reward = 0
        print("\n".join(summary))

env.close()
print(q_table)
print("\n".join(summary))

### Q learning - https:#medium.com/nerd-for-tech/q-learning-from-the-basics-b68e74f97254

# ACTION_SPACE = env.action_space.n # 3
# OBSERVATION_SPACE = len(env.observation_space.sample()) # continuous(2)
#
# Q_INCREMENTS = 2 # the number of discrete cells
# DISCRETE_OS_SIZE = [Q_INCREMENTS] * OBSERVATION_SPACE # an array of shape [20, 20]
# q_table = np.random.uniform(
#     low=-25, # low value is minimum reward
#     high=20, # high value is maximum reward
#     size=(DISCRETE_OS_SIZE + [ACTION_SPACE])) # an array of shape [20, 20, 3]
# # print("Qtable")
# # print(q_table)
#
#
# #? build a function that takes an observation and return the action given by the q table
# def obs_To_Index(env, obs, increments):
#
#     # get the bounds of the observation_space
#     obs_min = env.observation_space.low
#     obs_max = env.observation_space.high
#
#     # normalize the observation
#     obs = (obs - obs_min) / (obs_max - obs_min)
#
#     # convert the normalized array to an integer indice
#     indice = tuple(np.floor(obs * increments).astype(int))
#     # print(indice)
#
#     return indice
# #
#
# # store the initial state of the environment
# done = False
# observation = env.reset()
# while not done:
#     env.render()
#     # get the action coresponding to the current observation
#     indice = obs_To_Index(env, observation, Q_INCREMENTS)
#     print("Indice: {}".format(indice))
#     action = q_table[indice].argmax()
#     # take the action
#     new_observation, reward, done, info = env.step(action)
#     observation = new_observation
#     # print("New_observation: {}".format(new_observation))
# env.close()


# initialize parameters related to training
# LEARNING_RATE = 0.22048 # 0.1 # lr - how quickly values in the q table change
# DISCOUNT      = 0.95 # Y - how much the agent cares about future rewards
# LEARNING_RATE = 0.1 # 0.1 # lr - how quickly values in the q table change
# DISCOUNT      = 0.95 # Y - how much the agent cares about future rewards
# EPOCHS        = 400
# RENDER_EVERY  = 200   # how often to render a run

# calculate the predicted future reward
# new_indice = obs_To_Index(env, new_observation, Q_INCREMENTS)
# future_reward = reward + DISCOUNT * q_table[new_indice].max()
# # update the value in the q table
# current_q = q_table[indice + (action,)]
# new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * future_reward
#
# f = open("q_table_before.txt", "a")
# f.write(str(q_table))
# f.close()
#
# for e in range(EPOCHS):
#     # store the initial state of the environment
#     done = False
#     observation = env.reset()
#     while not done:
#         # render every [RENDER_EVERY] epochs
#         if e % RENDER_EVERY == 0:
#             env.render()
#
#         # get the action coresponding to the current observation
#         indice = obs_To_Index(env, observation, Q_INCREMENTS)
#         action = q_table[indice].argmax()
#         print(action)
#         # take the action
#         new_observation, reward, done, info = env.step(action)
#         # train the Q-Table
#         # calculate the predicted future reward
#         new_indice = obs_To_Index(env, new_observation, Q_INCREMENTS)
#         future_reward = reward + DISCOUNT * q_table[new_indice].max()
#         # update the value in the q table
#         current_q = q_table[indice + (action,)]
#         new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * future_reward
#         # update the observation
#         observation = new_observation
#
#         if done:
#             print(f'Reward {reward}')
#
#     # debug message
#     if e % 50 == 0:
#         print(f'Reached epoch {e}')
#
# f = open("q_table_after.txt", "a")
# f.write(str(q_table))
# f.close()
#
# env.close()


############## Deep learning - https:#www.section.io/engineering-education/building-a-reinforcement-learning-environment-using-openai-gym/

# states = len(env.observation_space.sample()) # continuous(2)
# actions = env.action_space.n
#
# def build_model(states, actions):
#     model = Sequential()
#     model.add(Dense(24, activation='relu', input_shape=states))
#     model.add(Dense(24, activation='relu'))
#     model.add(Dense(actions, activation='linear'))
#     return model
#
# model = build_model(states, actions)
# model.summary()


####### Simple Random
#
# episodes = 200  # 20 shower episodes
# RENDER_EVERY = 50
# for episode in range(1, episodes + 1):
#     state = env.reset()
#     done = False
#     score = 0
#
#     while not done:
#         # render every [RENDER_EVERY] epochs
#         # if episodes % RENDER_EVERY == 0:
#         #     env.render()
#
#         action = env.action_space.sample()
#         n_state, reward, done, info = env.step(action)
#         score += reward
#     print('Episode:{} Score:{}'.format(episode, score))


###### Q LEARNING - https:#www.kaggle.com/code/charel/learn-by-example-reinforcement-learning-with-gym/notebook#Markov-decision-process(MDP)

# NUM_ACTIONS = env.action_space.n
# NUM_STATES = env.observation_space.high
# # NUM_STATES = env.observation_space.n
# print(NUM_ACTIONS)
# print(NUM_STATES)
#
# Q = np.zeros([NUM_STATES, NUM_ACTIONS]) #You could also make this dynamic if you don't know all games states upfront
# gamma = 0.9 # discount factor
# alpha = 0.9 # learning rate
# for episode in range(1,1001):
#     done = False
#     rew_tot = 0
#     obs = env.reset()
#     while done != True:
#             action = np.argmax(Q[obs]) #choosing the action with the highest Q value
#             obs2, rew, done, info = env.step(action) #take the action
#             print(type(Q))
#             print(Q)
#             Q[obs,action] += alpha * (rew + gamma * np.max(Q[obs2]) - Q[obs,action]) #Update Q-marix using Bellman equation
#             #Q[obs,action] = rew + gamma * np.max(Q[obs2]) # same equation but with learning rate = 1 returns the basic Bellman equation
#             rew_tot = rew_tot + rew
#             obs = obs2
#     if episode % 50 == 0:
#         print('Episode {} Total Reward: {}'.format(episode,rew_tot))

# Q = np.zeros([NUM_STATES, NUM_ACTIONS])
# # env.observation.n, env.action_space.n gives number of states and action in env loaded
# # 2. Parameters of Q-learning
# eta = .628
# gma = .9
# epis = 5000
# rev_list = [] # rewards per episode calculate
# # 3. Q-learning Algorithm
# for i in range(epis):
#     # Reset environment
#     s = env.reset()
#     rAll = 0
#     d = False
#     j = 0
#     #The Q-Table learning algorithm
#     while j < 99:
#         env.render()
#         j+=1
#         # Choose action from Q table
#         a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
#         #Get new state & reward from environment
#         s1,r,d,_ = env.step(a)
#         #Update Q-Table with new knowledge
#         Q[s,a] = Q[s,a] + eta*(r + gma*np.max(Q[s1,:]) - Q[s,a])
#         rAll += r
#         s = s1
#         if d == True:
#             break
#     rev_list.append(rAll)
#     env.render()
# print("Reward Sum on all episodes " + str(sum(rev_list)/epis))
# print("Final Values Q-Table")
# print(Q)


######## Run Loops - Random Walker

# for i_episode in range(1):
#     observation = env.reset()
#     for t in range(50):
#         env.render()
#         # print(observation)
#         action = env.action_space.sample()
#         # print("ACTION: {}".format(action))
#
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             print("Reward is: {}".format(reward))
#             break
# env.close()



### Next approach - https:#github.com/kumarnikhil936/q_learning_mountain_car_openai/blob/master/mountaincar_q_learning.ipynb


# DISCRETE_OBSERVATION_SPACE_SIZE = [
#     20] * len(env.observation_space.high)  # will give out 20*20 list
#
# # see how big is the range for each of the 20 different buckets
# discrete_os_win_size = (env.observation_space.high -
#                         env.observation_space.low) / DISCRETE_OBSERVATION_SPACE_SIZE
#
# LEARNING_RATE = 0.1
# DISCOUNT = 0.95  # how important we find the new future actions are ; future reward over current reward
# EPISODES = 20000
# render = False
#
# # even though the solution might have been found, we still wish to look for other solutions
# epsilon = 0.5  # 0-1 ; higher it is, more likely for it to perform something random action
# START_EPSILON_DECAYING = 1
# # python2 style division - gives only int values
# END_EPSILON_DECAYING = EPISODES # 2
# epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)
#
# # Q learning
# # so we will have now a table such that each row will have 400 (20*20) rows for the possible state the agent can be in
# # and 3 columns for the 3 possible actions
# # the agent will see which state it is in and take the action corresponding to the largest Q value
#
# # Create a randomised q_table and agent will update it after exploring the environment
# q_table = np.random.uniform(
#     low=-2, high=0, size=(DISCRETE_OBSERVATION_SPACE_SIZE + [env.action_space.n]))
#
# # how to set low and high limits of rewards ? - if you see the rewards printed in below cell, they are mostly -1 and
# # might be something +ve only when you reach goal. Needs tweaking and playing around
# # print(q_table.shape)
#
#
# def get_discrete_state(state):
#     discrete_state = (state - env.observation_space.low) / discrete_os_win_size
#     return tuple(discrete_state.astype(np.int))  # return as tuple
#
# for ep in range(EPISODES):
#     done = False
#     discrete_state = get_discrete_state(env.reset())  # initial discrete state
#
#     if ep % 500 == 0:
#         render = True
#     else:
#         render = False
#         env.close()
#
#
#     while not done:  # goal reached means reward = 0
#
#         if np.random.random() > epsilon:
#             # in this environment, 0 means push the car left, 1 means to do nothing, 2 means to push it right
#             action = np.argmax(q_table[discrete_state])
#         else:
#             action = np.random.randint(0, env.action_space.n)
#
#         # Run one timestep of the environment's dynamics;  returns a tuple (observation, reward, done, info).
#         new_state, reward, done, _ = env.step(action)
#
#         new_discrete_state = get_discrete_state(new_state)
#
#         if render:
#             env.render()
#
#         if not done:
#             # max q value for the next state calculated above
#             max_future_q = np.max(q_table[new_discrete_state])
#
#             # q value for the current action and state
#             current_q = q_table[discrete_state + (action, )]
#
#             new_q = (1 - LEARNING_RATE) * current_q + \
#                 LEARNING_RATE * (reward + DISCOUNT * max_future_q)
#
#             # based on the new q, we update the current Q value
#             q_table[discrete_state + (action, )] = new_q
#
#         # goal reached; reward = 0 and no more negative
#         elif new_state[0] >= env.goal_position:
#             print(("Goal reached at {} episode".format(ep)))
#             # its like a snowbal effect, once the goal is reached - it will most likely reach again soon enough
#             q_table[discrete_state + (action, )] = 0
#
#         discrete_state = new_discrete_state
#
#     if END_EPSILON_DECAYING >= ep >= START_EPSILON_DECAYING:
#         epsilon -= epsilon_decay_value
#
# env.close()

