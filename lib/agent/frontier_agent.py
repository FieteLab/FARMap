import numpy as np
import torch
from torch import nn

from .memory_agent import MemoryAgent


class FrontierAgent(MemoryAgent):
    """
        Frontier-Based Exploration (Yamauchi, 1997)
    """
    def __init__(self, image_shape=None, decaying_factor=0.9, one_step=False, stochastic=False,
            is_planning=False, size_threshold=0, fragmentation=False, th_remap=0.3, th_lower=None, postpone=1,
            th_forget=0.00, use_rrt=False, use_memory_list=True, surp4frag='stm', update_first=True, epsilon=5, rho=2.0, frag_mode='z', unknown_marker=ord('X')):

        super(FrontierAgent, self).__init__(image_shape, decaying_factor, stochastic=stochastic, is_planning=is_planning, fragmentation=fragmentation, use_rrt=use_rrt, epsilon=epsilon, use_memory_list=False, rho=rho, frag_mode=frag_mode, unknown_marker=unknown_marker)
        self.head_bias = fragmentation
        self.use_size = fragmentation

    def step(self, observation, d, drop=False):
        """
            Given the current observation and a (global) head direction d, return the subgoal location (h, w, d).
        """
        candidates = observation.sum(1) == 0 # extract empty / occupied information. [0, 0, 0]: empty, o.w., occupied.
        if not candidates[:, -1, candidates.shape[-1]//2]: # agent is located on the wall
            breakpoint()
        candidates[:, -1, candidates.shape[-1]//2] = False
        if self.prev_action is None:  # set the default action
            self.prev_action = (0, 0, d)
        for i, obs in enumerate(observation):
            location = None
            if self.fragmentation:
                location = self.memory_step(observation, d, not drop, is_frontier=True, no_ltm_goal=True)
            else:
                self.stm_update(obs, self.prev_action, False)
            if candidates.sum() == 0:
                location = self.turn_back(d)
                break
            if type(location) is int and location == 2147483647:
                return location
            if location is None: # the current STM is the most desirable one.
                location = self.frontier_step(d, self.head_bias, self.use_size) # frontier-based exploration.


                if location is None: # there is no place to go within the current STM.
                    # force to convert to another
                    if self.fragmentation:
                        print("Force")
                        location = self.predict_from_STMgraph(d, force=True)
                        if location is None:
                            location = self.turn_back(d)
                    else:
                        location = self.turn_back(d)

        self.prev_action = location # set subgoal as previous action (it is used after excuting a plan toward the subgoal).
        return location
