#########
# Description: This code is usedto complete one full loop of a VANET Gym enviorment.
#
# Helper Docs:
#
# #www.section.io/engineering-education/building-a-reinforcement-learning-environment-using-openai-gym/
#########
# Author: Phillip Garrad

import gym
import json
from v2v_sim.envs.V2VSimulation import V2VSimulationEnv

## Load a scenario Configuration File
with open('./scenarios/scenario2.json') as json_file:
    data = json.load(json_file)

env = gym.make("V2VSimulation-v0")
env = V2VSimulationEnv(data)
env.reset()
env.render()

## Run the simulation for one action
action = 0
rew_tot=0
obs= env.reset()
env.render()
done = False
step = 0
while not done:
    print(f"Step: {step}")
    obs, rew, done, info = env.step(0)
    print(obs)
    print(rew)
    rew_tot = rew_tot + rew
    env.render()
    step += 1
#Print the reward of the set action
env.render()
rew_tot = rew_tot + rew
print(f"Step: {step}")
print("Reward: %r" % rew_tot)
