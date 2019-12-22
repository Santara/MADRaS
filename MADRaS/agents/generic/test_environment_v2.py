import numpy as np
import gym
from MADRaS.envs.gym_madras_v2 import MadrasEnv
import os


def test_madras_vanilla():
    env = MadrasEnv()
    print("Testing reset...")
    obs = env.reset()
    print("Initial observation: {}."
          " Verify if the number of dimensions {} is right.".format(obs, len(obs)))
    print("Testing step...")
    for t in range(20000):
        obs, r, done, _ = env.step({"MadrasAgent_0": [0.0, 1.0, -1.0]})
        print("{}: reward={}, done={}".format(t, r, done))
        dones = [x for x in done.values()]
        if np.any(dones):
            env.reset()
    os.system("pkill torcs")


def test_madras_pid():
    env = MadrasEnv()
    print("Testing reset...")
    obs = env.reset()
    print("Initial observation: {}."
          " Verify if the number of dimensions {} is right.".format(obs, len(obs)))
    print("Testing step...")
    for t in range(2000):
        obs, r, done, _ = env.step({"MadrasAgent_0": [0.3, 0.5],
                                    "MadrasAgent_1": [-0.3, 1.0]
                                })
        #print("{}".format(obs))
        print("{}: reward={}, done={}".format(t, r, done))
        dones = [x for x in done.values()]
        if ((np.any(dones)) or (t > 0 and t % 100 == 0)):
            env.reset()
    os.system("pkill torcs")


if __name__=='__main__':
    # test_madras_vanilla()
    test_madras_pid()