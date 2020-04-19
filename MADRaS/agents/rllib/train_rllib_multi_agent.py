import ray
import gym
import argparse
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.tune.logger import pretty_print
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from MADRaS.envs.gym_madras_v2 import MadrasEnv
import logging
import numpy as np
logging.basicConfig(filename='Telemetry.log', level=logging.DEBUG)

class MadrasRllib(MultiAgentEnv, MadrasEnv):
    """
        MADRaS rllib Env wrapper.
    """
    def __init__(self, *args):
        MadrasEnv.__init__(self)
    
    def reset(self):
        return MadrasEnv.reset(self)

    def step(self, action_dict):
        return MadrasEnv.step(self, action_dict)

def on_episode_end(info):
    episode = info["episode"]
    rewards = episode.agent_rewards
    total_episode = episode.total_reward


    episode.custom_metrics["agent0/rew_2"] = rewards[('MadrasAgent_0', 'ppo_policy_0')]**2.0
    episode.custom_metrics["agent1/rew_2"] = rewards[('MadrasAgent_1', 'ppo_policy_1')]**2.0
    episode.custom_metrics["env_rew_2"] = total_episode**2.0

def on_sample_end(info):
    print(info.keys())
    sample = info["samples"]
    print(dir(sample))
    splits = sample.policy_batches['ppo_policy_0'].split_by_episode()
    print(len(splits))
    for split in splits:
        print("EPISODE= ",np.sum(split['rewards']))
    
    

parser = argparse.ArgumentParser()
parser.add_argument("--num-iters", type=int, default=300)

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    env = MadrasRllib()

    obs_spaces, action_spaces = [], []
    for agent in env.agents:
        obs_spaces.append(env.agents[agent].observation_space)
        action_spaces.append(env.agents[agent].action_space)
    
    print(obs_spaces)
    print(action_spaces)
    policies = {"ppo_policy_{}".format(i) : (PPOTFPolicy, obs_spaces[i], action_spaces[i], {}) for i in range(env.num_agents)}

    def policy_mapping_fn(agent_id):
        id = agent_id.split("_")[-1]
        return "ppo_policy_{}".format(id)

    ppo_trainer = PPOTrainer(
        env=MadrasRllib,
        config={
            "eager": False,
            "num_workers": 1,
            "num_gpus": 0,
            "vf_clip_param": 20,
            # "sample_batch_size": 20, #set them accordingly
            "train_batch_size": 500,
            "callbacks": {
                "on_episode_end": on_episode_end,
                #"on_sample_end": on_sample_end,
            },
            #"lr": 5e-6,
            # "sgd_minibatch_size": 24,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
            },
        })

    #ppo_trainer.restore("{restore path}")

    for i in range(args.num_iters):
        print("== Iteration", i, "==")

        # improve the PPO policy
        if (i % 10 == 0):
           checkpoint = ppo_trainer.save()
           print("checkpoint saved at", checkpoint)
        
        logging.warning("-- PPO --")
        print(pretty_print(ppo_trainer.train()))
