import ray
import gym
import argparse
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from ray.tune.logger import pretty_print
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from MADRaS.envs.gym_madras_v2 import MadrasEnv


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

parser = argparse.ArgumentParser()
parser.add_argument("--num-iters", type=int, default=20)

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    env = MadrasRllib()

    obs_spaces, action_spaces = [], []
    for agent in env.agents:
        obs_spaces.append(env.agents[agent].observation_space)
        action_spaces.append(env.agents[agent].action_space)

    policies = {"ppo_policy_{}".format(i) : (PPOTFPolicy, obs_spaces[i], action_spaces[i], {}) for i in range(env.num_agents)}

    def policy_mapping_fn(agent_id):
        id = agent_id.split("_")[-1]
        return "ppo_policy_{}".format(id)

    ppo_trainer = PPOTrainer(
        env=MadrasRllib,
        config={
            "sample_batch_size": 50, #set them accordingly
            "train_batch_size": 100,
            "sgd_minibatch_size": 64,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
            },
        })

    for i in range(args.num_iters):
        print("== Iteration", i, "==")

        # improve the PPO policy
        print("-- PPO --")
        print(pretty_print(ppo_trainer.train()))
