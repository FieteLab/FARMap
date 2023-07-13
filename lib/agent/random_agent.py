import numpy as np
import torch
from torch import nn
from lib.planning import DP, RRT

class RandomAgent():
    def __init__(self, image_shape=None, is_planning=False, one_step=False, use_rrt=False):
        super(RandomAgent, self).__init__()
        self.image_shape = image_shape
        self.one_step = one_step
        self.is_planning = is_planning
        self.plans = []
        self.use_rrt = use_rrt
        self.rel_head_converter = np.asarray([
            [0, 1, 2, 3, 4],
            [1, 0, 3, 2, 4],
            [2, 3, 1, 0, 4],
            [3, 2, 0, 1, 4]])

        self.reverse_rel_head_converter = np.asarray([
            [0, 1, 2, 3, 4],
            [1, 0, 3, 2, 4],
            [3, 2, 0, 1, 4],
            [2, 3, 1, 0, 4]])



    def delta_converter(self, location, prev_d, rel_d=False):
        d, dh, dw = location
        dw = dw - (self.image_shape[-1]//2)
        dh = dh - (self.image_shape[-2]-1)
        if prev_d == 1:
            dh = -dh
            dw = -dw
        if prev_d == 2:
            temp = dh
            dh = -dw
            dw = temp
        if prev_d == 3:
            temp = dh
            dh = dw
            dw = -temp
        if type(dh) is torch.Tensor:
            dh = dh.item()
        if type(dw) is torch.Tensor:
            dw = dw.item()
        if type(d) is torch.Tensor:
            d = d.item()
        if rel_d:
            d = self.rel_head_converter[prev_d][d]
        return dh, dw, d


    def reverse_delta_converter(self, location, prev_d, rel_d=False):
        dh, dw, d = location
        if prev_d == 1:
            dh = -dh
            dw = -dw
        if prev_d == 2:
            temp = dh
            dh = dw
            dw = -temp
        if prev_d == 3:
            temp = dh
            dh = -dw
            dw = temp
        dw = dw + (self.image_shape[-1]//2)
        dh = dh + (self.image_shape[-2]-1)
        if type(dh) is torch.Tensor:
            dh = dh.item()
        if type(dw) is torch.Tensor:
            dw = dw.item()
        if type(d) is torch.Tensor:
            d = d.item()
        if rel_d:
            d = self.reverse_rel_head_converter[prev_d][d]
        return dh, dw, d




    def turn_back(self, d):
        print("TURN")
        next_d = [1, 0, 3, 2][d]
        h = [1, -1, 0, 0][d]
        w = [0, 0, 1, -1][d]
        return h, w, next_d


    def _planning(self, occupancy_map, current, subgoal, dist_map):
        if self.use_rrt:
            rrt_subgoals, _ = RRT(occupancy_map, current[:2], subgoal)
            if rrt_subgoals is None:
                return None, None
            rrt_subgoals = rrt_subgoals[1:-1]
            paths = [list(current)]
            h, w, d = current
            for rrt_subgoal in rrt_subgoals:
                dh = rrt_subgoal[0] - h
                dw = rrt_subgoal[1] - w
                if dh < 0:
                    dd = 0
                elif dh > 0:
                    dd = 1
                elif dw < 0:
                    dd = 2
                elif dw > 0:
                    dd = 3
                paths.append([int(rrt_subgoal[0]), int(rrt_subgoal[1]), dd])
            rrt_subgoal = paths[-1]
            path, dist_map = DP(occupancy_map, rrt_subgoal, subgoal, dist_map)
            if path is not None:
                paths += path[1:]
            else:
                return None, dist_map

        else:
            paths, dist_map = DP(occupancy_map, current, subgoal, dist_map)
        return paths, dist_map



    def planning(self, location, observation, prev_d, rel_d=False, dist_map=None):
        dh, dw, d = location
        degree = [0, 2, 1, 3][prev_d]
        observation = torch.rot90(observation, degree, dims=(-2, -1)).clone()
        if prev_d == 0:
            start = [self.image_shape[-2]-1, self.image_shape[-1]//2, prev_d]
        elif prev_d == 1:
            start = [0, self.image_shape[-1]//2, prev_d]
        elif prev_d == 2:
            start = [self.image_shape[-2]//2, self.image_shape[-1]-1, prev_d]
        elif prev_d == 3:
            start = [self.image_shape[-2]//2, 0, prev_d]

        occupancy_map = observation[0].sum(0) == 0

        if type(d) is torch.Tensor:
            d = d.item()

        current = start
        subgoal = [current[0]+dh, current[1]+dw, d]
        paths, dist_map = self._planning(occupancy_map, current, subgoal, dist_map)
        new_paths = []
        if paths is None:
            return None, dist_map
        for path in paths:
            h, w, d = path
            location = (h-current[0], w-current[1], d)
            new_paths.append(location)
        paths = new_paths[1:-1]
        return new_paths[1:-1], dist_map

    def step(self, observation, d):
        # find possible location
        candidates = observation.sum(1) == 0
        locations = []
        candidates[:,-1, 7] = False
        for candidate in candidates:
            location = candidate.nonzero()
            # only current location
            if len(location) == 1:
                location = self.turn_back(d)
            elif self.one_step:
                next_d = torch.randint(4, (1,))[0]
                dh = [-1, +1, 0, 0][next_d]
                dw = [0, 0, -1, +1][next_d]
                location = (dh, dw, next_d.item())
            else:
                order = torch.randperm(len(location))
                flag = False
                for idx in order:
                    if flag:
                        break
                    loc = location[idx]
                    for next_d in torch.randperm(4):
                        dh, dw = loc
                        locs = self.delta_converter((next_d, dh, dw), d)
                        if self.is_planning:
                            self.plans, dist_map = self.planning(locs, observation, d, rel_d=True)
                            if self.plans is not None:
                                flag = True
                                break
                        else:
                            flag = True
                            break
                if flag:
                    location = locs
                else:
                    print("IMPOSSIBLE")
                    location = self.turn_back(d)


#                location = self.delta_converter((next_d, dh, dw), d)
            locations.append(location)
        return location

    def plan(self):
        plan = self.plans
        self.plans = []
        if plan is None:
            return []
        return plan

    def memory_update(self, observations, plans):
        pass 
