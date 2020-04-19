import numpy as np
import gym
from MADRaS.envs.gym_madras_v2 import MadrasEnv
import os
import sys
import logging

logging.basicConfig(filename='Telemetry.log', level=logging.DEBUG)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# fh = logging.FileHandler('Telemetry.log')
# fh.setLevel(logging.DEBUG)
# logger.addHandler(fh)


def test_madras_vanilla():
    env = MadrasEnv()
    print("Testing reset...")
    obs = env.reset()
    print("Initial observation: {}."
          " Verify if the number of dimensions {} is right.".format(obs, len(obs)))
    print("Testing step...")
    a = [0.0, 1.0, -1.0]
    a = [0.0, 0.2, 0.0]
    b = [0.1, 0.3, 0.0]
    c = [0.2, 0.4, 0.0]
    for t in range(4000):
        obs, r, done, _ = env.step({"MadrasAgent_0": a, "MadrasAgent_1": b, "MadrasAgent_2": c})
        if ((t+1)%150 == 0):
            a = [0.0, -1.0, 1.0]
        print("{}: reward={}, done={}".format(t, r, done))
        dones = [x for x in done.values()]
        if np.any(dones):
            env.reset()
    os.system("pkill torcs")


def test_madras_pid():
    env = MadrasEnv()
    for key, val in env.agents.items():
        print("Observation Space ", val.observation_space)
        print("Obs_dim ", val.obs_dim)
    print("Testing reset...")
    obs = env.reset()
    a = [0.0, 0.2]
    b = [0.1, 0.00]
    c = [0.2, -0.2]
    print("Initial observation: {}."
          " Verify if the number of dimensions is right.".format(obs))
    for key, value in obs.items():
        print("{}: {}".format(key, len(value)))
    print("Testing step...")
    running_rew = 0
    for t in range(4000):
        obs, r, done, _ = env.step({"MadrasAgent_0": a, "MadrasAgent_1": b, "MadrasAgent_2": c})
        #print("{}".format(obs))
        #if ((t+1)%15 == 0):
        #    a = [0.0, 0.0]
        running_rew += r["MadrasAgent_0"]
        #print("{}: reward={}, done={}".format(t, running_rew, done))
        #logger.info("HELLO")
        if (done['__all__']):
            env.reset()
    os.system("pkill torcs")


if __name__=='__main__':
    test_madras_vanilla()
    #test_madras_pid()