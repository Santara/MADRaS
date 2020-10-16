"""
Gym Madras Env Wrapper.

This is an OpenAI gym environment wrapper for the MADRaS simulator. For more information on the OpenAI Gym interface, please refer to: https://gym.openai.com

Built on top of gym_torcs https://github.com/ugo-nama-kun/gym_torcs/blob/master/gym_torcs.py

The following enhancements were made for Multi-agent synchronization using exception handling:
- All the agents connect to the same TORCS engine through UDP ports
- If an agent fails to connect to the TORCS engine, it keeps on trying in a loop until successful
- Restart the episode for all the agents when any one of the learning agents terminates its episode

"""

import math
from copy import deepcopy
import numpy as np
import MADRaS.utils.snakeoil3_gym_v2 as snakeoil3
from MADRaS.utils.gym_torcs_v2 import TorcsEnv
from MADRaS.controllers.pid import PIDController
import gym
from gym.utils import seeding
from gym import spaces
import os
import subprocess
import signal
import time
from mpi4py import MPI
import socket
import MADRaS.utils.config_parser as config_parser
import MADRaS.utils.torcs_server_config as torcs_config
import MADRaS.utils.reward_handler as rm
import MADRaS.utils.done_handler_v2 as dm
import MADRaS.utils.observation_handler as om
import MADRaS.utils.communications_buffer as cb 
import MADRaS.utils.madras_datatypes as md
import MADRaS.traffic.traffic as traffic
from collections import OrderedDict
import multiprocessing
import logging

mt = md.MadrasDatatypes()
#logger = logging.getLogger(__name__)
path_and_file = os.path.realpath(__file__)
path, file = os.path.split(path_and_file)
DEFAULT_SIM_OPTIONS_FILE = os.path.join(path, "data", "sim_options_v2.yml")
DEFAULT_COMMUNICATION_MAP = os.path.join(path, "data", "communications.yml")


class MadrasAgent(TorcsEnv, gym.Env):
    def __init__(self, name, torcs_server_port, cfg, sim_info={}):
        self._config = config_parser.MadrasAgentConfig()
        self.name = name
        for key in sim_info:
            cfg[key] = sim_info[key]
        self._config.update(cfg)
        self.torcs_server_port = torcs_server_port
        TorcsEnv.__init__(self,
                          name=self.name,
                          vision=self._config.vision,
                          throttle=self._config.throttle,
                          gear_change=self._config.gear_change)

        if self._config.pid_assist:
            self.PID_controller = PIDController(self._config.pid_settings)
            self.action_dim = 2  # LanePos, Velocity
        else:
            self.action_dim = 3  # Steer, Accel, Brake
        self.observation_handler = om.ObservationHandler(self._config.observations,
                                                         self._config.vision)
        if self._config.normalize_actions:
            self.action_space = gym.spaces.Box(low=-np.ones(self.action_dim), high=np.ones(self.action_dim))

        self.observation_space = self.observation_handler.get_observation_space()
        self.obs_dim = self.observation_handler.get_state_dim()  # No. of sensors input
        self.state_dim = self.observation_handler.get_state_dim()
        self.reward_handler = rm.RewardHandler(self._config.rewards)
        self.done_handler = dm.DoneHandler(self._config.dones, self.name)
        self.initial_reset = True
        self.step_num = 0
        self.prev_dist = 0
        self.prev_damage = 0
        self.prev_pos = np.inf
        self.multi_flag = self._config.observations['multi_flag']
        self.buff_size = self._config.observations['buff_size']

    def create_new_client(self):
        while True:
            try:
                self.client = snakeoil3.Client(p=self.torcs_server_port,
                                                name=self.name)
                # Open new UDP in vtorcs
                self.client.MAX_STEPS = self._config.client_max_steps
                break
            except Exception as e:
                print("{} received error {} during client creation.".format(self.name, e))
                pass

    def get_observation_from_server(self):
        self.client.get_servers_input(0)
        # Get the initial input from torcs
        raw_ob = self.client.S.d
        # Get the current full-observation from torcs
        self.ob = self.make_observation(raw_ob)

        if np.any(np.asarray(self.ob.track) < 0):
            print("Reset produced bad track values.")
        self.distance_traversed = 0
        s_t = self.observation_handler.get_obs(self.ob, self._config)

        return s_t

    def complete_reset(self):
        if self._config.pid_assist:
            self.PID_controller.reset()
        self.reward_handler.reset()
        self.done_handler.reset()
        self.step_num = 0
        logging.info("Reset: Starting new episode")


    def reset_new(self, return_dict={}):
        self.create_new_client()
        return_dict[self.name] = self.get_observation_from_server()
        self.complete_reset()
        return return_dict

    def reset(self, return_dict={}):
        if self.initial_reset:
            self.wait_for_observation()
            self.initial_reset = False
        else:
            while True:
                try:
                    self.ob, self.client = TorcsEnv.reset(self, client=self.client, relaunch=True)
                except Exception:
                    self.wait_for_observation()

                if not np.any(np.asarray(self.ob.track) < 0):
                    break
                else:
                    print("Reset: Reset failed as agent started off track. Retrying...")

        if np.any(np.asarray(self.ob.track) < 0):
            print("Reset produced bad track values.")
        self.distance_traversed = 0
        print("Initial observation: {}".format(self.ob))
        s_t = self.observation_handler.get_obs(self.ob, self._config)
        if self._config.pid_assist:
            self.PID_controller.reset()
        self.reward_handler.reset()
        self.done_handler.reset()
        return_dict[self.name] = s_t

        return return_dict

    def wait_for_observation(self):
        """Refresh client and wait for a valid observation to come in."""
        self.ob = None
        while self.ob is None:
            print("{} Still waiting for observation".format(self.name))
            try:
                self.client = snakeoil3.Client(p=self.torcs_server_port,
                                               name=self.name)
                # Open new UDP in vtorcs
                self.client.MAX_STEPS = self._config.client_max_steps
                self.client.get_servers_input(0)
                # Get the initial input from torcs
                raw_ob = self.client.S.d
                # Get the current full-observation from torcs
                self.ob = self.make_observation(raw_ob)
                #self.client.respond_to_server()
            except:
                pass

    def step_vanilla(self, action):
        """Execute single step with steer, acceleration, brake controls."""
        if self._config.normalize_actions:
            action[1] = (action[1] + 1) / 2.0  # acceleration back to [0, 1]
            action[2] = (action[2] + 1) / 2.0  # brake back to [0, 1]
        r = 0.0
        try:
            self.ob, r, done, info = TorcsEnv.step(self, 0,
                                                   self.client, action,
                                                   self._config.early_stop)
        
        except Exception as e:
            print("Exception {} caught at port {}".format(str(e), self.torcs_server_port))
            self.wait_for_observation()
            #  TODO(santara): step not performed...

        game_state = {"torcs_reward": r,
                      "torcs_done": done,
                      "prev_distance_traversed": self.prev_dist,
                      "prev_racePos": self.prev_pos,
                      "prev_damage": self.prev_damage,
                      "distance_traversed": self.client.S.d['distRaced'],
                      "angle": self.client.S.d["angle"],
                      "damage": self.client.S.d["damage"],
                      "trackPos": self.client.S.d["trackPos"],
                      "track": self.client.S.d["track"],
                      "racePos": self.client.S.d["racePos"],
                      "num_steps": self.step_num}
        logging.info("[{}] Lane_pos:{} Distance:{} Speed{}".format(self.name,
                                                                    self.client.S.d["trackPos"],
                                                                    self.client.S.d['distRaced'],
                                                                    self.ob.speedX*self.default_speed))

        reward = self.reward_handler.get_reward(self._config, game_state)

        done = self.done_handler.get_done_signal(self._config, game_state)

        next_obs, full_obs = self.observation_handler.get_obs(self.ob, self._config, full_flag=True)

        info['prev_dist'] = game_state["distance_traversed"]
        info['prev_damage'] = game_state["damage"]
        info['prev_pos'] = game_state["racePos"]
        info['full_obs'] = full_obs._asdict()
        return next_obs, reward, done, info


    def step_pid(self, desire, e):
        """Execute single step with lane_pos, velocity controls."""
        if self._config.normalize_actions:
            # [-1, 1] should correspond to [-self._config.target_speed,
            #                                self._config.target_speed]
            speed_scale = self._config.target_speed
            desire[1] *= speed_scale  # Now in m/s
            # convert to km/hr
            desire[1] *= 3600/1000  # Now in km/hr
            # Normalize to gym_torcs specs
            desire[1] /= self.default_speed

        reward = 0.0

        #import pdb; pdb.set_trace()

        for PID_step in range(self._config.pid_settings['pid_latency']):

            a_t = self.PID_controller.get_action(desire)
            try:
                self.ob, r, done, info = TorcsEnv.step(self, PID_step,
                                                       self.client, a_t,
                                                       self._config.early_stop)
            except Exception as e:
                print("Exception {} caught at port {}".format(str(e), self.torcs_server_port))
                self.wait_for_observation()

            game_state = {"torcs_reward": r,
                          "torcs_done": done,
                          "prev_racePos": self.prev_pos,
                          "prev_damage": self.prev_damage,
                          "prev_distance_traversed": self.prev_dist,
                          "distance_traversed": self.client.S.d["distRaced"],
                          "angle": self.client.S.d["angle"],
                          "damage": self.client.S.d["damage"],
                          "trackPos": self.client.S.d["trackPos"],
                          "track": self.client.S.d["track"],
                          "racePos": self.client.S.d["racePos"],
                          "num_steps": self.step_num}
           
            reward += self.reward_handler.get_reward(self._config, game_state)
            self.prev_dist = game_state["distance_traversed"]
            self.prev_damage = game_state["prev_damage"]
            self.prev_pos = game_state["racePos"]
            
            if self._config.pid_assist:
                self.PID_controller.update(self.ob)
            done = self.done_handler.get_done_signal(self._config, game_state)
            info['prev_dist'] = self.prev_dist
            info['prev_damage'] = game_state["damage"]
            info['prev_pos'] = game_state["racePos"]
            if done:
                e.set()
                break
            if e.is_set():
                print("[{}]: Stopping agent because some other agent has hit done".format(self.name))
                break

        next_obs, full_obs = self.observation_handler.get_obs(self.ob, self._config, full_flag=True)
        
        info['full_obs'] = full_obs._asdict()
        return next_obs, reward, done, info

    def step(self, action, e, return_dict={}):
        if self._config.add_noise_to_actions:
            noise = np.random.normal(scale=self._config.action_noise_std, size=self.action_dim)
            action += noise
        if self._config.pid_assist:
            return_dict[self.name] = self.step_pid(action, e)
        else:
            return_dict[self.name] = self.step_vanilla(action)
        return return_dict

    def increment_(self, dist, damage, pos):
        self.prev_dist = dist
        self.prev_damage = damage
        self.prev_pos = pos
        #print("DIST_UPDATED: ", self.prev_dist)

    def increment_step(self):
        if self._config.pid_assist:
            self.step_num += self._config.pid_settings['pid_latency']
        else:
            self.step_num += 1
        #self.increment_dist()

    def comms_init(self, info={}):
        
        
        additional_vars_high = []
        additional_vars_low = []
        additional_dims = 0

        for var in info["vars"]:
            
            if (var == 'track'):
                additional_vars_high += 19*[1]
                additional_vars_low += 19*[0]
                additional_dims += 19
            elif (var == 'trackPos'):
                additional_vars_high += [1]
                additional_vars_low += [-1]
                additional_dims += 1
            elif (var == 'opponents'):
                additional_vars_high += 36*[1]
                additional_vars_low += 36*[0]
                additional_dims += 36
            elif ('speed' in var): #for handling speedX, speedY & speedZ
                additional_vars_high += [np.inf]
                additional_vars_low += [-np.inf]
                additional_dims += 1
            elif (var == 'action'):
                additional_vars_high += [1]*self.action_dim
                additional_vars_low += [-1]*self.action_dim
                additional_dims += self.action_dim
            else:
                additional_vars_high += [1]
                additional_vars_low += [-1]
                additional_dims += 1

        logging.info("NAME:{}, lencomms:{}".format(self.name, len(info['comms'])))
        self.obs_dim += additional_dims*len(info['comms'])
        self.comms_buffer = cb.CommBuffer(self.name, self.buff_size, additional_dims*len(info['comms']))
        additional_vars_high = additional_vars_high*len(info['comms'])
        additional_vars_low = additional_vars_low*len(info['comms']) 
        high = np.hstack((self.observation_space.high, additional_vars_high))
        low = np.hstack((self.observation_space.low, additional_vars_low))
        self.observation_space = spaces.Box(high=high, low=low)
        logging.info("{}: {}".format(self.name, self._config.observations['buff_size']))




class MadrasEnv(gym.Env):
    """Definition of the Gym Madras Environment."""
    def __init__(self, cfg_path=DEFAULT_SIM_OPTIONS_FILE):
        # If `visualise` is set to False torcs simulator will run in headless mode
        self._config = config_parser.MadrasEnvConfig()
        self._config.update(config_parser.parse_yaml(cfg_path))
        self.communications_map = dict()
        self.torcs_proc = None
        self.seed()
        
        self.num_agents = len(self._config.agents)
        self.agents = OrderedDict()
        self.init_wait = self._config.init_wait
        self.torcs_server_config = torcs_config.TorcsConfig(
            self._config.server_config, num_learning_cars= self.num_agents, randomize=self._config.randomize_env)
        self.start_torcs_process()
        if self._config.traffic:
            self.traffic_handler = traffic.MadrasTrafficHandler(
                self._config.torcs_server_port, len(self.agents), self._config.traffic) 
        
        num_traffic_agents = len(self._config.traffic) if self._config.traffic else 0
        
        if self._config.agents:
            for i, agent in enumerate(self._config.agents):
                agent_name = [x for x in agent.keys()][0]
                
                agent_cfg = agent[agent_name]
                name = '{}_{}'.format(agent_name, i)
                torcs_server_port = self._config.torcs_server_port + i + self.torcs_server_config.num_traffic_cars
                self.agents[name] = MadrasAgent(name, torcs_server_port, agent_cfg,
                                                {"track_len": self._config.track_len,
                                                 "max_steps": self._config.max_steps
                                                })
            
            if os.path.isfile(DEFAULT_COMMUNICATION_MAP):
                self.communications_map = config_parser.parse_yaml(DEFAULT_COMMUNICATION_MAP)
            
            print("COMMUNICATIONS MAP", self.communications_map)
            for comm_agent, agent_attrs in self.communications_map.items():
                self.agents[comm_agent].comms_init(agent_attrs)
        
           
        # TODO(santara): Can I not have different action dims for different agents?
        self.action_space = self.agents["MadrasAgent_0"].action_space
        self.observation_space = self.agents["MadrasAgent_0"].observation_space

        self.initial_reset = True
        print("Madras agents are: ", self.agents)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def config(self):
        return self._config
        
    def test_torcs_server_port(self):
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            udp.bind(('', self._config.torcs_server_port))
        except:
            print("Specified torcs_server_port {} is not available. "
                  "Searching for alternative...".format(
                  self._config.torcs_server_port))
            udp.bind(('', 0))
            _, self._config.torcs_server_port = udp.getsockname()
            print("torcs_server_port has been reassigned to {}".format(
                   self._config.torcs_server_port))

        udp.close()

    def start_torcs_process(self):
        #import pdb; pdb.set_trace()
        if self.torcs_proc is not None:
            os.killpg(os.getpgid(self.torcs_proc.pid), signal.SIGKILL)
            time.sleep(0.5)
            self.torcs_proc = None
        self.test_torcs_server_port()
        self.execute_torcs_launch_command()
        time.sleep(1)

    def reset_torcs(self):
        print("Relaunching TORCS on port{}".format(self._config.torcs_server_port))
        # os.kill(self.torcs_proc.pid, signal.SIGKILL)
        os.killpg(os.getpgid(self.torcs_proc.pid), signal.SIGKILL)
        time.sleep(1)
        self.execute_torcs_launch_command()
        time.sleep(1)

    def execute_torcs_launch_command(self):
        command = None
        rank = MPI.COMM_WORLD.Get_rank()
        self.torcs_server_config.generate_torcs_server_config()
        if rank < self._config.no_of_visualisations and self._config.visualise:
            command = 'export TORCS_PORT={} && torcs -t 10000000 -nolaptime'.format(
                       self._config.torcs_server_port)
        else:
            command = 'export TORCS_PORT={} && torcs -t 10000000 -r ~/.torcs/config/raceman/quickrace.xml -nolaptime'.format(self._config.torcs_server_port)
        if self._config.vision is True:
            command += ' -vision'

        self.torcs_proc = subprocess.Popen([command], shell=True, preexec_fn=os.setsid)
        print("TORCS server PID is: ", self.torcs_proc.pid)

    def reset_wait(self):
        stop_action = {}
        for agent_name, agent in self.agents.items():
            if (agent.action_dim == 3):
                stop_action[agent_name] = [0.0, -1.0, 1.0]
            else:
                stop_action[agent_name] = [0.0, 0.0]
        
        s_t = None
        for _ in range(self.init_wait):
            next_obs, _, _, _ = self.step(stop_action)
            s_t = next_obs

        return s_t
        

    def reset(self):
        """Reset Method to be called at the end of each episode."""
        #import pdb; pdb.set_trace()
        if not self.initial_reset:
            self.reset_torcs()
            self.torcs_server_config.generate_torcs_server_config()
        else:
            self.initial_reset = False

        if self._config.traffic:
            self.traffic_handler.reset(self.torcs_server_config.num_traffic_cars)
       
        s_t = OrderedDict()

        # Create clients and connect their sockets 
        for i, agent in enumerate(self.agents):
            self.agents[agent].torcs_server_port = self._config.torcs_server_port + i + self.torcs_server_config.num_traffic_cars
            self.agents[agent].create_new_client()

        # Collect first observations
        for agent in self.agents:
            self.agents[agent].prev_dist = 0.0
            s_t[agent] = self.agents[agent].get_observation_from_server()
            self.agents[agent].client.respond_to_server() # To elimate 10s of timeout error, responding to the server after obs
            self.agents[agent].increment_(self.agents[agent].client.S.d["distRaced"], self.agents[agent].client.S.d["damage"], self.agents[agent].client.S.d["racePos"]) 
        
        # Finish reset
        for agent in self.agents:
            self.agents[agent].complete_reset()

        for comm_agent in self.communications_map.keys(): #reset buffers
            self.agents[comm_agent].comms_buffer.reset()
            s_t[comm_agent] = np.hstack((s_t[comm_agent], self.agents[comm_agent].comms_buffer.request()))

        s_t = self.reset_wait()
        
        return s_t

    def step(self, action):
        next_obs, reward, done, info = {}, {}, {'__all__': False}, {}
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        e = multiprocessing.Event()
        jobs = []
        for agent in self.agents.keys():
            p = multiprocessing.Process(target=self.agents[agent].step, args=(action[agent], e, return_dict))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()
        
        done_check = False
        for agent in self.agents:
            next_obs[agent] = return_dict[agent][0]
            reward[agent] = return_dict[agent][1]
            done[agent] = return_dict[agent][2]
            if (done[agent] == True): 
                """
                    Although rllib supports individual agent resets
                    but MADRaS Env as of now has to be reset even if 
                    one of the agents hits done.
                """
                done_check = True
            info[agent] = return_dict[agent][3]
            self.agents[agent].increment_step()
            self.agents[agent].increment_(info[agent]['prev_dist'], info[agent]['prev_damage'], info[agent]['prev_pos'])

        for agent, agent_info in self.communications_map.items():
            full_obs_list = []
            for agent_c in agent_info["comms"]:
                full_obs_list.append((info[agent_c]["full_obs"], action[agent_c]))
            self.agents[agent].comms_buffer.insert(full_obs_list, agent_info["vars"]) 
        
        done['__all__'] = done_check

        for comm_agent in self.communications_map.keys():
            next_obs[comm_agent] = np.hstack((next_obs[comm_agent], self.agents[comm_agent].comms_buffer.request()))

        return next_obs, reward, done, info
