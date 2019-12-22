import numpy as np
import math
import warnings


class DoneManager(object):
    """Composes the done function from a given done configuration."""
    def __init__(self, cfg, agent_id):
        self.agent_id = agent_id
        self.dones = {}
        for key in cfg:
            try:
                exec("self.dones['{}'] = {}()".format(key, key))
            except:
                raise ValueError("Unknown done class {}".format(key))

        if not self.dones:
            warnings.warn("No done function specified. Setting TorcsDone "
                          "as done.")
            self.dones['TorcsDone'] = TorcsDone()

    def get_done_signal(self, game_config, game_state):
        done_signals = []
        
        for key, done_function in self.dones.items():
            done_val = done_function.check_done(game_config, game_state)
            done_signals.append(done_val)
            if done_val:
                if hasattr(done_function, "num_steps"):
                    done_function.reason = done_function.reason.format(done_function.num_steps) 
                out = "[{}] {}".format(self.agent_id, done_function.reason)
                print(out)
        
        signal = np.any(done_signals)
        return signal

    def reset(self):
        for done_function in self.dones.values():
            done_function.reset()


class MadrasDone(object):
    """Base class of MADRaS done function classes.
    Any new done class must inherit this class and implement
    the following methods:
        - [required] check_done(game_config, game_state)
        - [optional] reset()
    """
    def __init__(self, reason):
        self.reason = reason
        pass

    def check_done(self, game_config, game_state):
        del game_config, game_state
        raise NotImplementedError("Successor class must implement this method.")

    def reset(self):
        pass


class TorcsDone(MadrasDone):
    """Vanilla done function provided by TORCS."""

    def __init__(self):
        MadrasDone.__init__(self, "Done: Torcs Done")

    def check_done(self, game_config, game_state):
        del game_config
        if not math.isnan(game_state["torcs_done"]):
            return game_state["torcs_done"]
        else:
            return True


class RaceOver(MadrasDone):
    """Terminates episode when the agent has finishes one lap."""
    def __init__(self):
        MadrasDone.__init__(self, "Done: Race over!")

    def check_done(self, game_config, game_state):
        if game_state["distance_traversed"] >= game_config.track_len:
            return True
        else:
            return False


class TimeOut(MadrasDone):

    def __init__(self):
        MadrasDone.__init__(self, "Done: Episode terminated due to timeout.")

    def check_done(self, game_config, game_state):
        self.num_steps = game_state["num_steps"]
        if not game_config.max_steps:
            max_steps = int(game_config.track_len / game_config.target_speed * 50)
        else:
            max_steps = game_config.max_steps
        if self.num_steps >= max_steps:
            return True
        else:
            return False


class Collision(MadrasDone):
    
    def __init__(self):
        self.damage = 0.0
        MadrasDone.__init__(self, "Done: Episode terminated because agent collided.")

    def check_done(self, game_config, game_state):
        del game_config
        if self.damage < game_state["damage"]:
            self.damage = 0.0
            return True
        else:
            return False

    def reset(self):
        self.damage = 0.0


class TurnBackward(MadrasDone):

    def __init__(self):
        MadrasDone.__init__(self, "Done: Episode terminated because agent turned backward.")

    def check_done(self, game_config, game_state):
        del game_config
        if np.cos(game_state["angle"]) < 0:
            return True
        else:
            return False


class OutOfTrack(MadrasDone):

    def __init__(self):
        MadrasDone.__init__(self, "Done: Episode terminated because agent went out of track"
                  " after {} steps.")

    def check_done(self, game_config, game_state):
        self.num_steps = game_state["num_steps"]
        if (game_state["trackPos"] < -1 or game_state["trackPos"] > 1
            or np.any(np.asarray(game_state["track"]) < 0)):
            return True
        else:
            return False


class Rank1(MadrasDone):

    def __init__(self):
        MadrasDone.__init__(self, "Done: Episode terminated because agent is Rank 1"
                  " after {} steps.")

    def check_done(self, game_config, game_state):
        self.num_steps = game_state["num_steps"]
        if game_state["racePos"] == 1:
            return True
        else:
            return False