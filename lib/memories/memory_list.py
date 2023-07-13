import numpy as np
import copy

from .ltm import _prepare_merge_discrete
from lib.utils import check_discard

class MemoryList:
    """
        Lazy evaluation of calculating confidence map from multiple STMs which are possible to generated if recall has an error.
    """
    def __init__(self, stm=None):
        self.memories = []
        if stm is not None:
            self.memories.append(stm)
        self._current = None
        self.map = None

    @property
    def current(self):
        if self._current is not None:
            return self._current
        return self.memories[0].current

    @property
    def id(self):
        return self.memories[0].id

    @property
    def pixel_value(self):
        return self.memories[0].pixel_value

    @property
    def loc_diff_stms(self):
        if len(self.memories) == 1:
            return self.memories[0].loc_diff_stms

        o_curr = self.memories[0].current
        loc_diff_stms = self.memories[0].loc_diff_stms
        if self.current is not None:
            curr = self.current
            loc_diff_stms = loc_diff_stms.copy()
            delta = [curr[0] - o_curr[0], curr[1] - o_curr[1]]
            for k, v in loc_diff_stms.items():
                loc_diff_stms[k] = [v[0] + delta[0], v[1] + delta[1]]

        return loc_diff_stms


    def get_size(self):
        return self.memories[0].get_size()

    def check_passing_in_plan(self, plans):
        return self.memories[0].check_passing_in_plan(plans)
    
    def is_passing(self, d):
        return self.memories[0].is_passing(d)

    def drop(self):
        """
            drop all recalled memories.
        """
        self.memories = self.memories[:1]
        self._current = None
        self.map = None


    def add(self, stm):
        """
            Add new STM into the memory list if there is no contradiction.
        """
        if len(self.memories) == 0:
            self.memories.append(stm)
        else:
            # change pixel_value (pixel dictionary)
            stm = copy.deepcopy(stm)
            self.memories, pixel_values = _prepare_merge_discrete(self.memories + [stm])
            self.check_validity()

    
    def check_validity(self, target=None):
        """
            Check whether there is a contradiction between memories.
        """
        memories = []
        # every STM should have same pixel_value dictionary
        pixel_value = self.memories[0].pixel_value
        useful = []
        for pixel_key, idx in pixel_value.items():
            pixel = np.frombuffer(pixel_key, dtype=self.memories[0].pixel_dtype)
            if not np.all(pixel == self.memories[0].unknown_marker/255):
                useful.append(idx)

        if target is None:
            # first STM is the original STM
            memories = self.memories[:1]
            for i, stm in enumerate(self.memories[1:]):
                stm_map = stm.map[useful]
                if np.all((stm_map > 0).sum(0) <= 1):
                    memories.append(stm)
            self.memories = memories
        else:
            target = target[useful]
            # some memory does not match with the original
            return np.all((target > 0).sum(0) <= 1)



    def update_by_rel(self, state, rel_position, ignore_current=True, pos=None):
        self.map = None
        self._current = None

        for stm in self.memories:
            stm.update_by_rel(state, rel_position, ignore_current, pos)

        if len(self.memories) > 1:
            self.check_validity()
#            self.map, self._current = self._merge()
        else:
            self.map = self.memories[0].map
            self._current = self.memories[0].current
    

    def frontier_goals(self, option):
        return self.memories[0].frontier_goals(option)


    def get_occupancy(self):
        if len(self.memories) == 1:
            return self.memories[0].get_occupancy()

        entities = [np.stack(stm.get_occupancy()) for stm in self.memories]
        entity, current = self._merge(entities)
        self._current = current 
        frontier, empty, wall = entity
        return frontier, empty, wall


    def get_map(self, action=None, head=None, dh=0, dw=0, state_shape=(64,5,5), cache=True):
        if len(self.memories) == 1:
            ret_map = self.memories[0].get_map(action, head, dh, dw, state_shape)
            self._current = self.memories[0].current
            return ret_map
        
        if head is not None or action is not None:
            # generate merged graph
            maps = [stm.get_map(action, head, dh, dw, state_shape) for stm in self.memories]
            maps = np.stack(maps).mean(0)
            return maps

        if self.map is not None and cache:
            return self.map
        self.map, self._current = self._merge()
        return self.map

    
    def get_confidence_map(self):
        stm_map = self.get_map(cache=False)
        indices = []
        for pixel_key, idx in self.memories[0].pixel_value.items():
            pixel = np.frombuffer(pixel_key, dtype=self.memories[0].pixel_dtype)
            if not check_discard(pixel, self.memories[0].unknown_marker):
                indices.append(idx)
        confidence_map = stm_map[indices].sum(0)

        return confidence_map


    def get_empty(self):
        if len(self.memories) == 1:
            return self.memories[0].get_empty()

        empties = [stm.get_empty() for stm in self.memories]
        empty, current = self._merge(empties)
        self._current = current

        return empty


    def _merge(self, maps=None):
        is_map = maps is None # merge map
        if is_map:
            maps = [stm.get_map() for stm in self.memories]
            N = len(self.memories[0].pixel_value)
        else: # others such as occupancy
            N = len(maps[0])

        y1, x1, y2, x2 = [], [], [], []
        for _map, stm in zip(maps, self.memories):
            row, col = stm.current
            shape = _map.shape
            y1.append(row)
            x1.append(col)
            y2.append(shape[-2] - row)
            x2.append(shape[-1] - col)
        max_x1 = max(x1)
        max_x2 = max(x2)
        max_y1 = max(y1)
        max_y2 = max(y2)
        ndim = len(shape)

        if ndim == 3:
            merged_map = np.zeros((N, max_y1 + max_y2, max_x1 + max_x2))
        else:
            merged_map = -np.ones((max_y1 + max_y2, max_x1 + max_x2))

        center = (max_y1, max_x1)
        observed_step = -np.ones(merged_map.shape[-2:])
        
        exclude = []

        for i, _map in enumerate(maps):
            # check mismatch with the original STM (memories[0])
            if ndim == 3:
                if N != len(_map):
                    _map = _map[_map.sum((1,2)) != 0]
                merged_map[:, center[0]-y1[i]:center[0]+y2[i], center[1]-x1[i]:center[1]+x2[i]] += _map
                sub_map = merged_map[:, center[0]-y1[i]:center[0]+y2[i], center[1]-x1[i]:center[1]+x2[i]]
            else:
                merged_map[center[0]-y1[i]:center[0]+y2[i], center[1]-x1[i]:center[1]+x2[i]] += _map
            if is_map and not self.check_validity(sub_map):
                # revoke addition
                sub_map -= _map
                # remove i-th memory
                exclude.append(i)

        memories = []
        if len(exclude) > 0:
            print("EXCLUDE", exclude)
            if exclude[0] == 0:
                breakpoint()
        for k, stm in enumerate(self.memories):
            if k not in exclude:
                memories.append(stm)
        self.memories = memories

        merged_map /= (len(maps) - len(exclude))
        current = list(center)

        return merged_map, current

