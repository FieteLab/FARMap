import numpy as np
import torch
import os
import sys
import copy
import pandas as pd

from .memory import MappingMemory



def _prepare_merge_discrete(stms):
    pixel_values = {}
    for stm in stms:
        for k, v in stm.pixel_value.items():
            if k not in pixel_values:
                pixel_values[k] = len(pixel_values)
    for stm in stms:
        stm.update_discretize(pixel_values)
        stm.pixel_value = pixel_values.copy()

    return stms, pixel_values


def merge(stms, discrete=True, absorb=False):
    """
        Merge multiple STMs into one.
    """
    y1, x1, y2, x2 = [], [], [], []
    for stm in stms:
        row, col = stm.current
        shape = stm.get_map().shape
        y1.append(row)
        x1.append(col)
        y2.append(shape[-2] - row)
        x2.append(shape[-1] - col)
    max_x1 = max(x1)
    max_x2 = max(x2)
    max_y1 = max(y1)
    max_y2 = max(y2)
    ndim = len(shape)

    if discrete:
        stms, pixel_values = _prepare_merge_discrete(stms)
    if ndim == 3:
        merged_map = np.zeros((len(pixel_values), max_y1 + max_y2, max_x1 + max_x2))
    else:
        merged_map = -np.ones((max_y1 + max_y2, max_x1 + max_x2))
    center = (max_y1, max_x1)
    observed_step = -np.ones(merged_map.shape[-2:])
    for i, stm in enumerate(stms):
        if ndim == 3:
            merged_map[:, center[0]-y1[i]:center[0]+y2[i], center[1]-x1[i]:center[1]+x2[i]] += stm.get_map()
#            observed_step[center[0]-y1[i]:center[0]+y2[i], center[1]-x1[i]:center[1]+x2[i]] += stm.observed_step
        else:
            raise NotImplementedError
    observed_step[merged_map.sum(0)>0] = 0
    merged_map /= len(stms)
    # convert map into STM
    
    if absorb:
        stm = stms[0]
    else:
        fixed_cur_pos = stms[0].fixed_cur_pos
        egocentric = stms[0].egocentric
        decaying_factor = stms[0].decaying_factor
        discrete = stms[0].discrete
        stm = MappingMemory(fixed_cur_pos=fixed_cur_pos, egocentric=egocentric,
                                decaying_factor=decaying_factor, discrete=discrete)
    stm.current = list(center)
    stm.observed_step = observed_step
    stm.prev_action = stms[0].prev_action
    stm.map = merged_map
    stm.pixel_value = pixel_values.copy()
    return stm


class LongTermMemory:
    """
    Memorize (state, direction, short-term memory)
    """
    def __init__(self, memory_size=-1, read_option=0, discrete=False, get_new_stm=None, epsilon=5):
        self.memory = {}
        self.read_option = read_option
        self.memory_size = memory_size
        self.current_usage = 0
        self.discrete = discrete
        self.pixel_values = {}
        self.cache = {}
        self.memory_list = []
        self.current_id = 0
        self.memory_connection = {}
        self.dist_map = {}
        self.get_new_stm = get_new_stm
        self.epsilon = epsilon
    

    @property 
    def size_stat(self):
        # Check size of each local map
        size_info = []
        for k, v in self.memory.items():
            memory = self._read(v[-1])
            C, H, W = memory.get_size()
            size_info.append(H*W)
        return sum(size_info), len(size_info), size_info

    def state2key(self, state):
        return state.reshape(-1).tobytes()


    def remove(self):
        """
        Remove one memory randomly
        """
        count = 0
        # pick random index
        target = np.random.randint(self.memory_size)

        for k, v in self.memory.items():
            l = len(v)
            if count + l > target:
                if l == 1:
                    del self.memory[k]
                else:
                    temp = v[target-count]
                    v = v[:target-count] + v[target-count+1:]
                    del temp
                break
            count += l
        self.current_usage -= 1


    def write(self, state, stm, score=0, prev_action=None, connect=True):
        print(stm, stm.id, score, prev_action)
        stm = copy.deepcopy(stm)
        key = self.state2key(state)
        self.cache.pop(key, None)

        if self.current_usage == self.memory_size:
            self.remove()
        if key in self.memory:
            state, stms = self.memory[key]
            self.memory[key] = (state, stms + [stm])
            idx = len(stms) + 1
        else:
            self.memory[key] = (state, [stm])
            idx = 1

        self.current_usage += 1

        for k, v in stm.pixel_value.items():
            if k not in self.pixel_values:
                self.pixel_values[k] = len(self.pixel_values)

        # return new memory
        if connect:
            if self.get_new_stm is not None:
                new_stm = self.get_new_stm(None)
            else:
                new_stm = None

            self.connect(stm, score, new_stm, is_new=True)
            if new_stm is not None:
                borders = stm.extract_border(new_stm.id)
                new_stm.build_border(stm.id, borders)
            return new_stm


    def update_memory_list(self, prev_stm, prev_score):
        flag = False
        for i in range(len(self.memory_list)):
            if self.memory_list[i][0] == prev_stm.id:
                flag = True
                self.memory_list[i] = (prev_stm.id, prev_score, prev_stm)
                break
        if not flag:
            self.memory_list.append((prev_stm.id, prev_score, prev_stm))


    def check_valid_connect(self, prev_stm, next_stm):
        p_id = prev_stm.id
        n_id = next_stm.id

        if p_id not in next_stm.loc_diff_stms or n_id not in prev_stm.loc_diff_stms:
            return True
        return False
        h, w = next_stm.current
        flag1 = False
        for loc in next_stm.loc_diff_stms[p_id]:
            if loc[0] == h and loc[1] == w:
                flag1 = True
        
        h, w = prev_stm.current
        flag2 = False
        for loc in prev_stm.loc_diff_stms[n_id]:
            if loc[0] == h and loc[1] == w:
                flag2 = True
        return not(flag1 and flag2)

    
    def connect_index(self, id1, id2):
        def _connect_index(id1, id2):
            if id1 in self.memory_connection:
                self.memory_connection[id1].add(id2)
            else:
                self.memory_connection[id1] = set([id2])

        _connect_index(id1, id2)
        _connect_index(id2, id1)



    def connect(self, prev_stm, prev_score, next_stm, is_new=False):
        """
            Connect previous STM and the current STM.
        """
        if not self.check_valid_connect(prev_stm, next_stm):
            return

        next_id = next_stm.id 
        if is_new:
            self.current_id += 1
            next_stm.connect(prev_stm.id)
            next_stm.id = self.current_id
            next_id = self.current_id
        else:
            breakpoint()

        prev_stm.connect(self.current_id)
        self.update_stm_distance_map(prev_stm)
        self.connect_index(self.current_id, prev_stm.id)

        # update memory_list
        self.update_memory_list(prev_stm, prev_score)


    def predict_from_STMgraph(self, stm, score=1, mem_ids=[], force=False):
        """
            Use connectivity graph for finding subgoal.
        """

        epsilon = self.epsilon
        if epsilon <= 0:
            min_dist = stm.frontier_goals()[1]
            if len(min_dist) > 0:
                min_dist = min_dist[0]
            else:
                min_dist = 0

        if len(self.memory_list) == 0:
            return None
        keys = []
        ids = []
        indices = []
        scores = []
        stms = []
        for id, s, memory in self.memory_list:
            ids.append(id)
            stms.append(memory)
            scores.append(s)
        
        current_id = stm.id
        score_vec = -np.ones((max(current_id, max(ids))+1))
        for id, s in zip(ids, scores):
            score_vec[id] = s
    
        score_vec[current_id] = score
        

        distance_matrix = self.get_stm_distance_matrix(stm)
        dist_vec = distance_matrix[stm.id]
        mask = dist_vec == 2147483647 # just big enough value
        mem_ids = pd.unique(mem_ids[::-1])
        if epsilon > 0:
            score = score_vec / (dist_vec + epsilon)
        else:
            dist_vec[stm.id]  = min_dist
            score = score_vec / dist_vec.clip(min=1e-5)

        # masking not directly connected one
        for mem_id in mem_ids:
            if mem_id not in stm.loc_diff_stms and mem_id != stm.id:
                mask[mem_id] = True

        
        for i, mem_id in enumerate(mem_ids):
            if mem_id == stm.id:
                temporal_penalty = 1
            else:
                temporal_penalty = i / len(mem_ids)  #min(1, 0.8 + 0.1 * i)
                temporal_penalty = min(1, 0.95 + 0.01 * i)

            score[mem_id] = score[mem_id] * temporal_penalty


        score[mask] = -1
        score[mask] = 0

        if force:
            score[current_id] = -1

        stm_id = score.argmax()

        if stm_id == current_id:
            return None

        print(stm.id, stm_id, mem_ids, score, score_vec, dist_vec)
        
        stm_id = self.memory_list[stm_id][0]
        print('score_map', score_vec, stm_id)
        locations = stm.loc_diff_stms[stm_id]
        current = stm.current.copy()

        # temporarilly stm_id and the index of list are the same.
        ret_stm = self.memory_list[stm_id][2]

        # potentially, if two STMs are connected with multiple times, it has a problem.
        dists = [stm.distance(current, l) for l in locations]
        idx = np.argmin(dists)
        loc = locations[idx]
        ret_loc = [loc[0] - current[0], loc[1] - current[1], loc[2]]

        # change current location to fragmented location
        if len(ret_stm.loc_diff_stms[stm.id]) > 1:
            check = False
            for loc in ret_stm.loc_diff_stms[stm.id]:
                if loc[0] == ret_stm.current[0] and loc[1] == ret_stm.current[1]:
                    check = True
                    break
            if not check:
                ret_stm.current = ret_stm.loc_diff_stms[stm.id][0].copy()[:2]
        else:
            ret_stm.current = ret_stm.loc_diff_stms[stm.id][0].copy()[:2]

        return ret_loc, ret_stm


    def update_stm_distance_map(self, stm):
        for node_i, v in stm.dist_to_diff_stms.items():
#            print(stm.dist_to_diff_stms)
            for node_j, dist in v.items():
                if node_i in self.dist_map:
                    self.dist_map[node_i][node_j] = min(dist, self.dist_map[node_i].get(node_j,2147483647))
                else:
                    self.dist_map[node_i] = {node_j: dist}
                if node_j in self.dist_map:
                    self.dist_map[node_j][node_i] = min(dist, self.dist_map[node_j].get(node_i,2147483647))
                else:
                    self.dist_map[node_j] = {node_i: dist}
    

    def get_stm_distance_matrix(self, stm):
        ids, dists, locs = stm.get_distance_to_other_stms()
        
        keys = self.dist_map.keys()
        N = 1 + max(list(keys)+ ids.tolist() + [stm.id])
        dist_matrix = np.ones((N,N)) * 2147483647

        current = stm.id
        for id in np.unique(ids):
            for d in dists[ids==id]:
                dist_matrix[current][id] = d
                dist_matrix[id][current] = d
              
        for i in range(N):
            dist_matrix[i,i] = 0
        for k in keys:
            if k == current:
                continue
            for j, d in self.dist_map[k].items():
                if j == current:
                    continue
                dist_matrix[k, j] = d
                dist_matrix[j, k] = d


        return dist_matrix


    def recall_passing(self, curr_stm_id, next_stm_id, diff):
        for stm_id, score, stm in self.memory_list:
            if stm_id == next_stm_id:
                break
        
        frag_locs = stm.loc_diff_stms[curr_stm_id]
        if len(frag_locs) > 1:
            # obs compare?
            breakpoint()

        h, w = frag_locs[0][:2]
        h = h + diff[0]
        w = w + diff[1]

        stm.current = [h, w]
        stm.check_range()
        return stm


    def read(self, state):
        key = self.state2key(state)
        if key not in self.memory:
            return None
        if key in self.cache:
            return self.cache[key]
        else:
            ret = self._read(self.memory[key][1])
            self.cache[key] = ret
            return ret


    def _read(self, stms):
        """
        Read Option:
                0: merge
                1: first-stored
                2: last-stored
                3: randomly pick
        """
        if self.read_option == 0:
            ret = merge(stms, self.discrete)
        elif self.read_option == 1:
            idx = np.random.randint(len(stms))
            ret = stms[idx]
        elif self.read_option == 2:
            ret = stms[0]
        elif self.read_option == 3:
            ret = stms[-1]
        else:
            raise NotImplementedError
        return ret


if __name__ == '__main__':

#    from .memory import MappingMemory
    # general merge test
    discrete=True
    get_mem = lambda x :MappingMemory(shape=(3,2,2), fixed_cur_pos=(4,2), egocentric=True, discrete=discrete)

    memory = get_mem(None)
    first_mem = memory
    ltm = LongTermMemory(discrete=True, get_new_stm=get_mem, read_option=3)
    states = []
    for action in [0,0,0,0,0]:
        state = np.random.rand(3,15,15)
        states.append(state)
#        memory.update(state, action)
        memory.update_by_rel(state, (-4, 3,0)) 
        memory = ltm.write(state, memory)

    print(ltm.get_stm_distance_matrix(first_mem))
    stm = ltm.read(states[2])
    print(stm.id)
    for action in [0, 0, 1]:
        state = np.random.rand(3,15,15)
        stm.update_by_rel(state, (-4,3,0))

    memory= ltm.write(state, stm)
    
    print(ltm.get_stm_distance_matrix(stm))
    breakpoint()

    state = np.arange(25).reshape(5,5)
    state = np.stack((state, state, state))

    for discrete in [True, False]:
        memory = MappingMemory(shape=(3,2,2), fixed_cur_pos=(4,2), egocentric=True, discrete=discrete)
        memory2 = MappingMemory(shape=(3,2,2), fixed_cur_pos=(4,2), egocentric=True, discrete=discrete)
        state2 = np.copy(state)
        ltm = LongTermMemory(discrete=discrete)
        for action in [None, 0, 1, 2, 1, 3, 1, 1, 1]:
            memory.update(state, action)
            state2[0,0, 0] = 127
            memory2.update(state2, action)
            print(memory2.pixel_value)
            ltm.write(state, memory)
            ltm.write(state, memory2)
            print(ltm.read(state))
            break

