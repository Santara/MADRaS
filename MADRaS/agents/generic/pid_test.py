import numpy as np
import gym
from MADRaS.envs.gym_madras_v2 import MadrasEnv
import os
import sys
import logging

logging.basicConfig(level=logging.DEBUG)

def test_madras_pid(vel, file_name):
    env = MadrasEnv()
    for key, val in env.agents.items():
        print("Observation Space ", val.observation_space)
        print("Obs_dim ", val.obs_dim)
    print("Testing reset...")
    obs = env.reset()
    vel = float(vel)
    a = [0.0, vel]
    b = [0.1, 0.00]
    c = [0.2, -0.2]
    print("Initial observation: {}."
          " Verify if the number of dimensions is right.".format(obs))
    for key, value in obs.items():
        print("{}: {}".format(key, len(value)))
    print("Testing step...")
    running_rew = 0
    speeds = []
    for t in range(300):
        obs, r, done, _ = env.step({"MadrasAgent_0": a})
        #print("{}".format(obs))
            
        #    a = [0.0, 0.0]
        running_rew += r["MadrasAgent_0"]
        #print("{}: reward={}, done={}".format(t, running_rew, done))
        #logger.info("HELLO")
        speeds.append(obs["MadrasAgent_0"][21])
        if (done['__all__']):
            env.reset()
    print(speeds)
    np.save(file_name, np.array(speeds))
    os.system("pkill torcs")


if __name__=='__main__':
    #test_madras_vanilla()
    test_madras_pid(sys.argv[1], sys.argv[2])