import numpy as np
import torch
import sys

def get_dist_map(dist_map, occupancy_map, goal, target):
    """
        Calculate Distance Map
            Input:
                dist_map: numpy array of shape (4, H, W) for distance map
                occupancy_map: numpy array of shape (H, W)
                goal: tuple of (h, w, d) for goal point
                target: list of (h, w, d) for the target points
            Output:
                dist_map: numpy array of shape (4, H, W) for distance map
    """
    H, W = occupancy_map.shape
    dhs = [-1, 1, 0, 0]
    dws = [0, 0, -1, 1]
    eh, ew, ed = goal
    while len(target)>0:
        h, w, d = target[0]
        target = target[1:]
        val = float(dist_map[d, h, w])
        if (dist_map == sys.maxsize)[:,occupancy_map!=0].sum() == 0:
            raise Exception("No Connection")
        for _d, dh, dw in zip(range(4), dhs, dws):
            _h = h + dh
            _w = w + dw
            if _h < 0 or _w < 0 or _h >= H or _w >= W:
                continue
            if occupancy_map[_h, _w] == 0: # blocking
                continue
            if dist_map[_d, _h, _w] > val + 1 and val + 1 <= dist_map[ed, eh, ew]:
                dist_map[_d, _h, _w] = val + 1
                target.append([_h, _w, _d])
    return dist_map


def DP(occupancy_map, st, goal, dist_map=None):
    """
        Use DP to find the shortest path from st to goal.
            Input:
                occupancy_map: numpy array of shape (H, W)
                st: tuple of (h, w, d) for start point
                goal: tuple of (h, w, d) for goal point
                dist_map: numpy array of shape (4, H, W) for distance map (if it is pre-computed)
            Output:
                path: list of (h, w, d) for the shortest path
                dist_map: numpy array of shape (4, H, W) for distance map
    """
    H, W = occupancy_map.shape
    flag = dist_map 
    if dist_map is None:
        if type(occupancy_map) is np.ndarray and False:
            dist_map = np.zeros((4, H, W)) + sys.maxsize
        else:
            dist_map = torch.zeros((4, H, W)) + sys.maxsize
        h ,w, d = st
        dist_map[d, h, w] = 0
    
    target = (dist_map < sys.maxsize).nonzero()
    target = torch.index_select(target, -1, torch.tensor([1,2,0])).tolist()

    eh, ew, ed = goal
    dhs = [-1, 1, 0, 0]
    dws = [0, 0, -1, 1]
    val = 0
    if dist_map[ed, eh, ew] == sys.maxsize:
        dist_map = get_dist_map(dist_map, occupancy_map, goal, target)
    
    paths = [goal]
    h, w, d = goal

    if dist_map[ed, eh, ew] == sys.maxsize:
        return None, dist_map
    val = float(dist_map[d, h, w])
    while val > 0:
        for _d in range(4):
            _h = h - dhs[d]
            _w = w - dws[d]
            if _h < 0 or _w < 0 or _h >= H or _w >= W:
                continue
            if dist_map[_d, _h, _w]  == val - 1:
                paths.append([_h, _w, _d])
                h, w, d = _h, _w, _d
                val = val - 1
                break
    return paths[::-1], dist_map
    


