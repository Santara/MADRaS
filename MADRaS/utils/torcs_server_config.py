import numpy as np
import random
import os
import logging
import MADRaS.utils.data.track_details as track_details
logger = logging.getLogger(__name__)

path_and_file = os.path.realpath(__file__)
path, _ = os.path.split(path_and_file)

QUICKRACE_TEMPLATE_PATH = os.path.join(path, "data", "quickrace.template")
CAR_CONFIG_TEMPATE_PATH = os.path.join(path, "data", "car_config.template")
SCR_SERVER_CONFIG_TEMPLATE_PATH = os.path.join(path, "data", "scr_server_config.template")

MAX_NUM_CARS = 10  # There are 10 scr-servers in the TORCS GUI

DIRT_TRACK_NAMES = [
    "dirt-1",
    "dirt-2",
    "dirt-3",
    "dirt-4",
    "dirt-5",
    "dirt-6",
    "mixed-1",
    "mixed-2"
]

ROAD_TRACK_NAMES = [
    "aalborg",
    "alpine-1",
    "alpine-2",
    "brondehach",
    "g-track-1",
    "g-track-2",
    "g-track-3",
    "corkscrew",
    "eroad",
    # "e-track-1", # Segmentation fault from torcs: no track observations after about 1000 steps
    "e-track-2",
    "e-track-3",
    "e-track-4",
    "e-track-6",
    "forza",
    "ole-road-1",
    "ruudskogen",
    "spring",
    "street-1",
    "wheel-1",
    "wheel-2"
]



class TorcsConfig(object):
    def __init__(self, cfg, num_learning_cars, randomize=False):
        self.max_cars = cfg["max_cars"] if "max_cars" in cfg else MAX_NUM_CARS
        self.min_traffic_cars = cfg["min_traffic_cars"] if "min_traffic_cars" in cfg else 0
        self.track_category = cfg["track_category"] if "track_category" in cfg else "road"
        self.num_learning_cars = num_learning_cars
        self.track_names = cfg["track_names"] if "track_names" in cfg else ROAD_TRACK_NAMES
        self.distance_to_start = cfg["distance_to_start"] if "distance_to_start" in cfg else 0
        self.torcs_server_config_dir = (cfg["torcs_server_config_dir"] if "torcs_server_config_dir" in cfg
                                        else "/home/sohan/.torcs/config/raceman/")
        self.scr_server_config_dir = (cfg["scr_server_config_dir"] if "scr_server_config_dir" in cfg
                                          else "/home/sohan/usr/local/share/games/torcs/drivers/scr_server/")
        with open(QUICKRACE_TEMPLATE_PATH, 'r') as f:
            self.quickrace_template = f.read()
        with open(CAR_CONFIG_TEMPATE_PATH, 'r') as f:
            self.car_config_template = f.read()
        with open(SCR_SERVER_CONFIG_TEMPLATE_PATH, 'r') as f:
            self.scr_server_config_template = f.read()
        self.randomize = randomize
        self.quickrace_xml_path = os.path.join(self.torcs_server_config_dir, "quickrace.xml")
        self.scr_server_xml_path = os.path.join(self.scr_server_config_dir, "scr_server.xml")
        self.traffic_car_type = cfg['traffic_car'] if 'traffic_car' in cfg else 'car1-trb1'
        self.learning_car_types = cfg['learning_car']
        self.track_length = 0
        self.track_width = 0

    def get_num_traffic_cars(self):
        if not self.randomize:
            return self.max_cars-self.num_learning_cars
        else:
            num_traffic_cars = np.random.randint(low=self.min_traffic_cars, high=self.max_cars)
            return num_traffic_cars

    def get_track_name(self):
        if not self.randomize:
            track_name = self.track_names[0]
        else:
            track_name = random.sample(self.track_names, 1)[0]
        logging.info("-------------------------CURRENT TRACK:{}------------------------".format(track_name))
        self.track_length = track_details.track_lengths[track_name]
        self.track_width = track_details.track_widths[track_name]
        return track_name

    def get_learning_car_type(self):
        if not self.randomize:
            learning_car_types = self.learning_car_types[:self.num_learning_cars]
        else:
            learning_car_types = random.sample(self.learning_car_types, self.num_learning_cars)
        for i, car in enumerate(learning_car_types):
            logging.info("-------------------------LEARNING CAR {}:{}------------------------".format(i, car))
        return learning_car_types

    def generate_torcs_server_config(self):
        self.num_traffic_cars = self.get_num_traffic_cars()
        logging.info("-----------------------Num. Traffic Cars:{}-----------------------".format(self.num_traffic_cars))

        car_config = "\n".join(self.car_config_template.format(
                      **{"section_no": i+1, "car_no": i}) for i in range(
                        self.num_traffic_cars + self.num_learning_cars))
        context = {
            "track_name": self.get_track_name(),
            "track_category": self.track_category,
            "distance_to_start": self.distance_to_start,
            "car_config": car_config
        }
        torcs_server_config = self.quickrace_template.format(**context)
        with open(self.quickrace_xml_path, "w") as f:
            f.write(torcs_server_config)
        self.learning_car_types = self.get_learning_car_type()
        car_name_list = ([self.traffic_car_type]*self.num_traffic_cars + self.learning_car_types +
                         [self.traffic_car_type]*(MAX_NUM_CARS-self.num_traffic_cars-self.num_learning_cars))
        scr_server_config = self.scr_server_config_template.format(*car_name_list)
        with open(self.scr_server_xml_path, "w") as f:
            f.write(scr_server_config)