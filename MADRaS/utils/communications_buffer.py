from queue import Queue 
from copy import deepcopy
import numpy as np
import MADRaS.utils.madras_datatypes as md

mt = md.MadrasDatatypes()

class CommBuffer:
    """
        To hold the last k actions of taken 
        by an agent. 
    """

    def __init__(self, agent_id, size, buffer_dim):
        self._agent_id = agent_id
        self._size = size
        self._buffer_dim = buffer_dim
        self._buffer = Queue(maxsize=self._size)
        self._curr_size = 0

    def insert(self, full_obs, var_list):
        
        if (self._curr_size < self._size):
            self._curr_size += 1
            self._buffer.put(self.parse_buffer_items(full_obs, var_list))
        else: 
            _ = self._buffer.get()
            self._buffer.put(self.parse_buffer_items(full_obs, var_list))

    def request(self):
        temp_queue = Queue(maxsize=self._size)
        ret = np.zeros((self._size*self._buffer_dim,), dtype=mt.floatX)
        for i in range(self._curr_size):
            agent_var = self._buffer.get()
            temp_queue.put(agent_var)
            ret[i*self._buffer_dim: (i+1)*self._buffer_dim] = agent_var
        self._buffer = temp_queue
        return ret

    def parse_buffer_items(self, full_obs_list, var_list):
        buffer_array = []
        for full_obs_act in full_obs_list:
            full_obs, action = full_obs_act
            for var in var_list:
                if var == 'action':
                    buffer_array.append(action)
                else:
                    val = full_obs[var]
                    buffer_array.append(val)
        buffer_array = np.hstack(buffer_array)
        return buffer_array

    def reset(self):
        del self._buffer
        self._buffer = Queue(maxsize=self._size)
        self._curr_size = 0
