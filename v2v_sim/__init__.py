from gym.envs.registration import register
import json

with open('./scenarios/scenario1.json') as json_file:
    data = json.load(json_file)

register(
    id='V2VSimulation-v0',
    entry_point='v2v_sim.envs:V2VSimulationEnv',
    kwargs={"data": data},
)