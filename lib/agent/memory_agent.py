import numpy as np
import copy
import torch
from torch import nn

from lib.memories import MappingMemory, LongTermMemory, MemoryList, merge
from lib.utils import remap_checker, get_score, check_discard
from .random_agent import RandomAgent

class MemoryAgent(RandomAgent):
    def __init__(self, image_shape=None, decaying_factor=0.9, one_step=False, stochastic=False,
            is_planning=False, fragmentation=False, th_remap=0.3, th_lower=0.1, postpone=1,
            th_forget=0.00, use_rrt=False, use_memory_list=True, surp4frag='stm', update_first=True, epsilon=5, rho=2, frag_mode='z', unknown_marker=ord('X')):
        super(MemoryAgent, self).__init__(image_shape, is_planning, one_step, use_rrt=use_rrt)
        self.stochastic = stochastic
        self.fragmentation = fragmentation
        self.unknown_marker = unknown_marker

        
        stm = MappingMemory(shape=image_shape, decaying_factor=decaying_factor, egocentric=True, discrete=True, fixed_cur_pos=(image_shape[-2]-1,image_shape[-1]//2), unknown_marker=unknown_marker)
        if use_memory_list:
            self.stm = MemoryList(stm)
        else:
            self.stm = stm

        # create a new map
        self.new_stm = lambda x : MappingMemory(shape=image_shape, decaying_factor=decaying_factor, egocentric=True, discrete=True, fixed_cur_pos=(image_shape[-2]-1,image_shape[-1]//2), unknown_marker=unknown_marker)

        # read option 0: merge, 1: first-stored, 2: last-stored, 3: randomly sampled
        self.ltm = LongTermMemory(discrete=True, get_new_stm=self.new_stm, read_option=2, epsilon=epsilon)
        self.prev_action = None
        self.scores = []
        self.th_remap = th_remap
        self.th_lower = th_lower
        self.postpone = postpone
        self.th_forget = th_forget
        self.remap_loc = []
        self.recall_loc = []
        self.ltm_subgoal_loc = []
        self.memory_size = []
        self.use_memory = True
        self.saved_stm = None
        self.curr_step = 0
        self.prev_surprisal = None
        self.frag_mode = frag_mode
        self.surp4frag = surp4frag
        self.surprisal_in_plan = None
        self.update_first = update_first # lazy evaluation for updating the observation, True: update first and act fragmentation/recall.
        self.rho = rho
        self.size_threshold = 3 # a threshold for the size of frontier to set a subgoal.
        self.mem_ids = []


    @property
    def stm_score(self):
        """
            Calculate a score q:
            q = # of frontier / # of empty
        """
        frontier, empty, wall = self.stm.get_occupancy()
        num_frontier = frontier.sum()
        num_empty = empty.sum()
        num_wall = wall.sum()
        score = num_frontier  / num_empty
        return score


    def get_confidence_map(self):
        ret_map = self.stm.get_confidence_map()
        curr = self.stm.current
        return ret_map, curr


    def get_stm(self):
        if type(self.stm) is MemoryList:
            stm = self.stm.memories[0]
        else:
            stm = self.stm
        return stm


    def stm_update(self, obs, action, ignore_current=False, update_mem_size=True):
        """
            Update the current STM given an observation.
        """
        self.mem_ids.append(self.stm.id)
        self.stm.update_by_rel(obs, action, ignore_current=ignore_current)
        if self.th_forget > 0:
            self.stm.forgetting(self.th_forget)
        C, H, W = self.stm.get_size()
        if update_mem_size:
            self.memory_size.append(H*W)


    def update_score(self, location, obs, s_map=None):
        """
            update current surprisal for future fragmentation.
        """
        h, w, d = location
        if self.update_first:
            h = w = 0

        score = 0
        if 'stm' in self.surp4frag:
            mem_map = self.stm.get_map(head=d, dh=h, dw=w, state_shape=(1000, self.image_shape[-2], self.image_shape[-1]))
            score = get_score(mem_map, obs, self.stm.pixel_value, self.unknown_marker)
        if 'sps' in self.surp4frag:
            if self.surprisal_in_plan is not None and len(self.surprisal_in_plan) > 0:
                s = self.surprisal_in_plan[0]
                self.surprisal_in_plan = self.surprisal_in_plan[1:]
            else:
                s = 0
            if '+' in self.surp4frag:
                score = (score + s) / 2
            else:
                score = s
        self.scores.append(score)



    def frontier_planning(self, location, prev_d, occupancy_map, rel_d=False, dist_map=None):
        """
            Planning over the current observation.
            It is used for planning for frontier-based exploration and LTM graph-based one.
        """
        dh, dw, d = location
        stm = self.get_stm()
        h, w = stm.current
        start = [h, w, prev_d]

        if type(d) is torch.Tensor:
            d = d.item()

        current = start
        subgoal = [current[0]+dh, current[1]+dw, d]
        # run planning and get a sequence of actions and distance map.
        paths, dist_map = self._planning(occupancy_map, current, subgoal, dist_map)
        new_paths = []
        if paths is None: # impossible to reach out that location.
            return None, dist_map
        for path in paths:
            h, w, d = path
            location = (h-current[0], w-current[1], d)
            new_paths.append(location)
        # ignore the first and the last plan since the first one will be taken in action right a way and the last one is the location of subgoal.
        return new_paths[1:-1], dist_map


    def memory_update(self, observations, plans):
        """
            update memory from a trajectories (planning).
        """
        if len(plans) == 0 or not self.use_memory:
            return
        prev_plan = plans[0]

        if self.update_first:
            self.stm_update(observations[0], prev_plan, False)
        if self.fragmentation:
            self.fragmentation_update(observations[0], prev_plan[0], prev_plan[1], prev_plan[2])
        if not self.update_first:
            self.stm_update(observations[0], prev_plan, False)

        # get confidence map
        rets = [self.get_confidence_map()]


        for obs, plan in zip(observations[1:], plans[1:]):
            dh = plan[0] - prev_plan[0]
            dw = plan[1] - prev_plan[1]

            if self.update_first:
                self.stm_update(obs, (dh, dw, plan[-1]), False)
            if self.fragmentation:
                self.fragmentation_update(obs, dh, dw, plan[-1])
            if not self.update_first:
                self.stm_update(obs, (dh, dw, plan[-1]), False)

            prev_plan = plan
            rets.append(self.get_confidence_map())

        dh = self.prev_action[0] - plans[-1][0]
        dw = self.prev_action[1] - plans[-1][1]
        self.prev_action = (dh, dw, self.prev_action[-1])
        return rets



    def predict(self, mem_map):
        """
            Predict surprisal (1 - averaged confidence) of given observation
        """
        pixel_value = self.stm.pixel_value
        valid_indices = []
        for i, pixel in enumerate(pixel_value):
            pixel = np.frombuffer(pixel, dtype=np.float32)
            if check_discard(pixel):
                valid_indices.append(i)
        if len(valid_indices) == 0:
            return 1
        observed = mem_map[valid_indices].max(0)
        score = observed.mean()
        return 1-score


    def recall(self, stm):
        """
            Attach recalled LTM into current STM
        """
        self.stm.add(stm)
        return


    def fragmentation_update(self, obs, h, w, d, s_map=None, is_frontier=False):
        """
            Check Recall and Fragmentation.
        """
        if type(obs) is torch.Tensor:
            obs = obs.cpu().numpy()
        is_passing = self.stm.is_passing(d) # check whether recall happens.
        recalled = False
        if is_passing is not None:
            stm_id, diff = is_passing
            stm = self.ltm.recall_passing(self.stm.id, stm_id, diff)
            recalled = stm is not None
            if recalled:
                print("passing", self.stm.id, stm.id)
                self.recall_loc.append(len(self.scores))
                is_new = self.stm.id not in [e[0] for e in self.ltm.memory_list]
                if is_new:
                    store_stm = self.stm.memories[0] if type(self.stm) is MemoryList else self.stm
                    self.ltm.write(obs[0], store_stm, self.stm_score, connect=False)
                    self.ltm.update_memory_list(store_stm, self.stm_score)
                # pass
                if type(self.stm) is MemoryList:
                    self.stm = MemoryList(stm)
                else:
                    self.stm = stm

        # update surprisal information
        self.update_score((h, w, d), obs, s_map)


        # check the fragmentation, if we just recalled, do not check it
        if self.saved_stm is None and not recalled and remap_checker(self.scores, self.th_remap, self.th_lower, T=self.postpone, remap_loc=self.remap_loc, dist=True, rho=self.rho, mode=self.frag_mode):
            print('frag!', len(self.scores)-1)
            self.remap_loc.append(len(self.scores)-1)
            # store the current stm in LTM.
            stm = self.get_stm()
            stm = self.ltm.write(obs, stm, self.stm_score)
            print(stm.id, stm.current, stm.loc_diff_stms, stm.map.shape)
            if stm is None:
                stm = self.new_stm(None)

            if type(self.stm) is MemoryList:
                self.stm = MemoryList(stm)
            else:
                self.stm = stm

            if is_frontier: # update the current observation to the newly created STM.
                self.stm_update(obs, (0,0,d), False, update_mem_size=False)


    def predict_from_STMgraph(self, d, force=False):
        """
            Find LTM sua plan toward the subgoall
        """

        stm = self.get_stm()
        ret = self.ltm.predict_from_STMgraph(stm, self.stm_score, self.mem_ids, force)
        if ret is None:
            return None
        goal, next_stm = ret
        # based on distance, choose the head direction
        dh, dw, dd = goal
        # set head direction ordering if the goal location is not reachable since the agent cannot rotate in place.
        if d == 0:
            ds = [1, 2, 3, 0]
        elif d == 1:
            ds = [0, 3, 2, 1]
        elif d == 2:
            ds = [3, 0, 1, 2]
        elif d == 3:
            ds = [2, 1, 0, 3]

        location = [goal[0], goal[1], ds[0]]
        if self.is_planning:
            dist_map = None
            empty = stm.get_empty()
            h, w = stm.current
            if empty[h + goal[0], w + goal[1]] == 0:
                empty[h+goal[0], w+goal[1]] = 1
            for _d in ds:
                location = [goal[0], goal[1], _d]
                self.plans, dist_map = self.frontier_planning(location, d, empty, dist_map=dist_map)
                if self.plans is not None:
                    break
            if self.plans is None: # problem happens
                print("PLAN FAIL")
                return None

        # set next_stm
        self.saved_stm = next_stm # this is the only case when saved_stm is not set to None.
        self.ltm_subgoal_loc.append(len(self.scores))
        return location 


    def memory_step(self, observation, d, update_stm=True, is_frontier=False, no_ltm_goal=False):
        """
            Process everything related to memory
        """
        # init
        if type(observation) is torch.Tensor:
            observation = observation.cpu().numpy()
        prev_action = self.prev_action

        if self.prev_action is None:
            self.prev_action = (0, 0, d)

        flag = self.saved_stm is None

        use_graph = False
        if update_stm and self.update_first:
            self.stm_update(observation[0], self.prev_action, False)


        # FarMap: use LTM
        if self.fragmentation and prev_action is not None:
            # See the last few lines of predict_from_STMgraph()
            if self.saved_stm is not None: # go to the most desirable STM which is not the current STM.
                if not self.update_first and update_stm:
                    self.stm_update(observation[0], self.prev_action, False)
                stm = self.get_stm() 
                print("SHIFT TO ANOTHER STMS {} -> {}".format(stm.id, self.saved_stm.id))
                self.saved_stm = None
                use_graph = True

            # check fragmentation.
            self.fragmentation_update(observation[0], prev_action[0], prev_action[1], d, is_frontier=is_frontier)


        if update_stm:
            if use_graph:
                self.stm_update(observation[0], (0,0,d), False, update_mem_size=False)
            elif not self.update_first:
                self.stm_update(observation[0], self.prev_action, False)

        if no_ltm_goal:
            score = self.stm_score
            use_ltm_goal = score < 0.05 # if surprisal is high, manually stay in the current STM.
        else:
            use_ltm_goal = True

        if use_ltm_goal and self.fragmentation and flag:
            return self.predict_from_STMgraph(d, force=False)
        return None


    def find_subgoal(self, d, head_bias, use_size):
        """
            Find Subgoal.
        """
        ret = self.stm.frontier_goals('nearest_centroid') # find subgoal candidates.
        if ret is None:
            return None
        subgoals, dist, sizes, (empty, unknown) = ret
        subgoals = subgoals[dist>0]
        sizes = sizes[dist>0]
        dist = dist[dist>0]
        if len(subgoals) == 0: # there is no subgoal in the current STM (e.g., already fully explored).
            return None
        useful = sizes >= min(self.size_threshold, sizes.max()) # minimum frontier-edge size.
        if useful.sum() > 0:
            subgoals = subgoals[useful]
            dist = dist[useful]
            sizes = sizes[useful]
            dist = 1/(dist+1e-6)
            if use_size:
                dist = dist * sizes
            p = dist / sum(dist)
            if head_bias:
                p = self.add_head_direction_bias(subgoals, p, d)
            # weighted sampling.
            if self.stochastic:
                idx = np.random.choice(np.arange(len(dist)), p=p)
            else:
                idx = 0
        else:
            idx = 0
        if len(subgoals) == 0:
            return None
        # choose subgoal.
        location = subgoals[idx]
        return location, empty


    def frontier_step(self, d, head_bias=False, use_size=False):
        """
            Find Subgoal and Planning.
        """
        ret = self.find_subgoal(d, head_bias, use_size)
        if ret is None:
            return None
        else:
            location, empty = ret

        if self.is_planning:
            dist_map = None
            self.plans, dist_map = self.frontier_planning(location, d, empty, dist_map=dist_map)

            if self.plans is None:
                h, w = self.stm.current
                hs, ws = empty.nonzero()
                hs -= h
                ws -= w
                dist = abs(hs-location[0]) + abs(ws-location[1])
                order = dist.argsort()
                hs = hs[order]
                ws = ws[order]
                dd = location[2]
                flag = False
                # once the location is set, we should also decide the head direction of subgoal.
                for _d in [dd, (dd+1)%4, (dd+2)%4, (dd+3)%4]:
                    if flag:
                        break
                    for _h, _w in zip(hs, ws):
                        candidate = (_h, _w, _d)
                        # if the current candidate is reachable, set it as the subgoal.
                        self.plans, dist_map = self.frontier_planning(candidate, d, empty, dist_map=dist_map)
                        if self.plans is not None:
                            location = candidate
                            flag = True
                            break

        # Fail to find subgoal.
        if location[0] == 0 and location[1] == 0 and len(self.plans) == 0:
            return 2147483647

        return location


    def add_head_direction_bias(self, subgoals, prob, d):
        """
            Add head direction constraint:
                Ignore centroids located in backward direction of agent (based on head direction).
        """
        if d ==0:
            target = subgoals[:, 0] > 0 
        elif d == 1:
            target = subgoals[:, 0] < 0
        elif d == 2:
            target = subgoals[:,1] > 0
        elif d == 3:
            target = subgoals[:,1] < 0

        prob[target] *= 1e-5
        prob /= sum(prob)

        return prob
