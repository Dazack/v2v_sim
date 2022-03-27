import gym # openAi gym
from gym import envs
import numpy as np
import datetime
import keras
import json
import tensorflow as tf
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

# Let's first do some random steps in the simulation so you see how the simulation looks like

# rew_tot=0
# obs= env.reset()
# env.render()
# for _ in range(6):
#     action = env.action_space.sample() #take step using random action from possible actions (actio_space)
#     obs, rew, done, info = env.step(action)
#     rew_tot = rew_tot + rew
#     env.render()
# #Print the reward of these random action
# print("Reward: %r" % rew_tot)




# # Let's see how the algorithm solves the taxi game by following the policy to take actions delivering max value
#
# rew_tot=0
# obs= env.reset()
# env.render()
# done=False
# while done != True:
#     action = np.argmax(Q[obs])
#     obs, rew, done, info = env.step(action) #take step using selected action
#     rew_tot = rew_tot + rew
#     env.render()
# #Print the reward of these actions
# print("Reward: %r" % rew_tot)



###### Q LEARNING

# Q = np.zeros([env.observation_space.n,env.action_space.n])
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


for i_episode in range(200):
    observation = env.reset()
    for t in range(100):
        # env.render()
        # print(observation)
        action = env.action_space.sample()
        # print("ACTION: {}".format(action))

        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print("Reward is: {}".format(reward))
            break
env.close()