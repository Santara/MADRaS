"""
Runs a constant velocity traffic agent along the center of a track
and generates a 2D map of the track. It then puts together a LUT mapping
distance from start of the race to an (x, y) coordinate of the corresponding
point of the road in the map.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from MADRaS.traffic.traffic import ConstVelTrafficAgent
import MADRaS.utils.torcs_server_config as torcs_config
import socket
import subprocess
import time
import logging
import logging.config
import sys
import pickle as pkl
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

MAPDIR = "/home/anirban/Projects/MADRaS_revisited/MADRaS_tmp/MADRaS/utils/data/maps"

class MappingAgentManager(object):
    def __init__(self, track_name, visualise=False):
        self.track_name = track_name
        self.server_cfg = {
            "max_cars": 1,
            "track_names": [self.track_name],
            "distance_to_start": 0,
            "learning_car": ['car1-trb1']
        }
        self.torcs_server_config = torcs_config.TorcsConfig(
            self.server_cfg, randomize=False)
        self.visualise = visualise
        self.find_free_udp_port()
        self.torcs_server_config.generate_torcs_server_config()
        self.start_torcs_server()
        self.mapping_agent = MadrasMappingAgent(self.torcs_server_port, track_length=self.torcs_server_config.track_length)

    def find_free_udp_port(self):
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.bind(('', 0))
        _, self.torcs_server_port = udp.getsockname()

    def start_torcs_server(self):
        if self.visualise:
            command = 'export TORCS_PORT={} && vglrun torcs -t 10000000 -nolaptime'.format(self.torcs_server_port)
        else:
            command = 'export TORCS_PORT={} && torcs -t 10000000 -r ~/.torcs/config/raceman/quickrace.xml -nolaptime'.format(self.torcs_server_port)

        self.torcs_proc = subprocess.Popen([command], shell=True, preexec_fn=os.setsid)
        time.sleep(1)

    @property
    def track_map(self):
        return self.mapping_agent.map

    def map(self):
        self.mapping_agent.map_track()
        return self.track_map

    def plot_map(self):
        if not self.track_map:
            raise ValueError("No track_map found. Run MappingAgentManager.map() to "
                             "generate track_map first.")
        else:
            coordinates = np.vstack([x[1] for x in self.track_map])
            plt.figure()
            plt.plot(coordinates[:, 0], coordinates[:, 1])
            # TODO(santara): turn off axes - no point in showing axes
            plt.savefig(os.path.join(MAPDIR, self.track_name+'.png'))
            # if self.visualise:
            plt.show()

    def save_map(self):
        with open(os.path.join(MAPDIR, self.track_name+'.pkl'), 'wb') as f:
            pkl.dump(self.track_map, f)
            f.close()


class MadrasMappingAgent(ConstVelTrafficAgent):
    def __init__(self, port, cfg=None, track_length=None):
        if cfg is None:
            cfg = {
                "target_speed": 50,
                "target_lane_pos": 0.0,
                "collision_time_window": 1,
                "pid_settings": {
                    "accel_pid": [10.5, 0.05, 2.8],
                    "steer_pid": [5.1, 0.001, 0.000001],
                    "accel_scale": 1.0,
                    "steer_scale": 0.1
                    }
            }
        super(MadrasMappingAgent, self).__init__(port, cfg, "MappingAgent")
        self.coordinates = np.zeros(2)
        self.heading = 0  # rad. angle w.r.t. horizontal
        self.map = [[0, self.coordinates]]
        self.track_length = track_length

    def get_coordinates(self):
        init_pos = self.coordinates
        init_vel = self.ob.speedX
        accel = self.action[1]
        delta_t = 1/50.
        delta_s = init_vel * delta_t + 0.5 * accel * delta_t ** 2
        delta_pos = [delta_s * np.cos(self.heading), delta_s * np.sin(self.heading)]
        final_pos = init_pos + delta_pos
        return final_pos

    def map_track(self):
        self.wait_for_observation()
        num_steps = 0
        while True:
            self.action = self.get_action()
            delta_heading = np.deg2rad(self.action[0] * 2.21)  # Found the number 2.21 through trial and error
            self.heading += delta_heading
            self.coordinates = self.get_coordinates()
            self.map.append([self.ob.distFromStart, self.coordinates])
            try:
                self.ob, _, done, _ = self.env.step(0, self.client, self.action)
            
            except Exception as e:
                logging.debug("Exception {} caught by {} traffic agent at port {}".format(
                                str(e), self.name, self.port))
                self.wait_for_observation()
            num_steps += 1
            if done or (self.ob.distRaced > self.track_length):
                break

            self.detect_and_prevent_imminent_crash_out_of_track()
            self.PID_controller.update(self.ob)


if __name__=="__main__":
    mapping_manager = MappingAgentManager('spring', False)
    map = mapping_manager.map()
    mapping_manager.plot_map()
    mapping_manager.save_map()