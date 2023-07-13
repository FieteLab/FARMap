import numpy as np
import torch

from lib.utils import masking, check_discard


class MappingMemory:
    def __init__(self, shape=(3,100,100), fixed_cur_pos=None, egocentric=False, decaying_factor=0, discrete=False, size_limit=(250,250), id=0, init_val_one=False, unknown_marker=ord('X')):
        # define starting coordinate
        self.step = 0
        self.unknown_marker = unknown_marker
        self.expansion_unit = list(shape)
        self.observed_step = -np.ones(shape[-2:])
        if fixed_cur_pos is None:
            self.current = [int(shape[0]/2), int(shape[1]/2)]
            self.start = [int(shape[0]/2), int(shape[1]/2)]
        else:
            self.current = list(fixed_cur_pos)
            self.start = list(fixed_cur_pos)
        self.fixed_cur_pos = fixed_cur_pos
        self.egocentric = egocentric
        self.prev_action = 0
        self.ndim = len(shape)
        self.action_converter = np.asarray([
            [0, 1, 2, 3, 4],
            [1, 0, 3, 2, 4],
            [2, 3, 1, 0, 4],
            [3, 2, 0, 1, 4]])

        self.pixel_value = {}
        self.pixel_dtype = np.uint8
        self.discrete = discrete
        self.map = -np.ones(shape)
        self.decaying_factor = decaying_factor

        # location of path to move to other STMs.
        self.loc_diff_stms = {}
        self.dist_to_diff_stms = {}

        self.size_limit = size_limit
        self.id = id
        self.borders = {}
        # starting confidence 1 or alpha
        if init_val_one:
            self.init_val = 1
        else:
            self.init_val = 1 - self.decaying_factor 
        if self.discrete:
            self.map += 1


    def get_confidence_map(self):
        stm_map = self.get_map()
        indices = []
        for pixel_key, idx in self.pixel_value.items():
            pixel = np.frombuffer(pixel_key, dtype=self.pixel_dtype)
            if not check_discard(pixel, self.unknown_marker):
                indices.append(idx)
        confidence_map = stm_map[indices].sum(0)
        return confidence_map


    def extract_border(self, stm_id):
        h, w = self.current
        indices = []
        d = self.prev_action
        H, W = self.map.shape[1:]
        if d <= 1:
            for i in reversed(range(w)):
                if self.map[0, h, i] > 0:
                    indices.append([0,i-w])
                else:
                    break
            for i in range(w, W):
                if self.map[0, h, i] > 0:
                    indices.append([0,i-w])
                else:
                    break
        else:
            for i in reversed(range(h)):
                if self.map[0, i, w] >0:
                    indices.append([i-h, 0])
                else:
                    break
            for i in range(h, H):
                if self.map[0, i, w] > 0:
                    indices.append([i-h, 0])
                else:
                    break
        indices = np.asarray(indices)
        _d = np.zeros((len(indices), 1)) + d
        indices = np.concatenate((indices, _d), axis=-1)
        self.build_border(stm_id, indices.copy())

        indices[:, 2] = [1, 0, 3, 2][d]

        return indices


    def is_passing(self, d):
        """
            Check whether the agent is currently passing the wall in STM
        """
        h, w = self.current
        for k, v in self.borders.items():
            for _h, _w, _d in v:
                if _h == h and _w == w and _d == d:
                    fh, fw, _ = self.loc_diff_stms[k][0]
                    delta = [int(_h - fh), int(_w - fw)]
                    return k, delta
        return None

    def check_passing_in_plan(self, plans):
        """
            Check whether the agent is passing the wall in STM in planned actions
        """
        if plans is None:
            return None
        h, w = self.current
        if len(self.borders) == 0:
            return plans
        borders = np.concatenate([v for v in self.borders.values()])
        for dh, dw, d in plans:
            cond = np.logical_and((borders[:,0] == h+dh), (borders[:,1] == w+dw))
            cond = np.logical_and(cond, borders[:,2] == d)
            if cond.sum() > 0:
                print('fail', plans)
                return None
        return plans


    def check_range(self):
        """
            Check the size of the map whether it needs to be expannded.
        """
        h, w = self.current
        while self.current[0] < 0:
            self.expansion(0)
        while self.current[0] >= self.map.shape[-2]:
            self.expansion(1)
        while self.current[1] < 0:
            self.expansion(2)
        while self.current[1] >= self.map.shape[-1]:
            self.expansion(3)


    def build_border(self, stm_id, border):
        """
            Add a border for recall that pass the fracture point.
        """
        border[:,0] = border[:,0] + self.current[0]
        border[:,1] = border[:,1] + self.current[1]
        if stm_id in self.borders:
            self.borders[stm_id] = np.concatenate((self.borders[stm_id], border))
        else:
            self.borders[stm_id] = border

        while self.borders[stm_id][:,0].min() < 0:
            self.expansion(0)
        while self.borders[stm_id][:,0].max() >= self.map.shape[-2]:
            self.expansion(1)
        while self.borders[stm_id][:,1].min() < 0:
            self.expansion(2)
        while self.borders[stm_id][:,1].max() >= self.map.shape[-1]:
            self.expansion(3)

        if self.borders[stm_id][:,0].min() < 0:
            breakpoint()


    def get_distance_to_other_stms(self):
        """
            Calculate the distance to other STM (d_cj).
        """
        ret_dists = []
        ret_locs = []
        ret_ids = []

        for k, v in self.loc_diff_stms.items():
            ret_dists += [self.distance(self.current, l) for l in v]
            ret_locs += v
            ret_ids += [k for _ in v]


        if len(self.loc_diff_stms) == 0:
            breakpoint()

        ret_dists = np.stack(ret_dists)
        ret_ids = np.stack(ret_ids)
        ret_locs = np.stack(ret_locs)

        return ret_ids, ret_dists, ret_locs




    def distance(self, loc1, loc2):
        """
            Return Manhattan Distance.
        """
        l1_dist = abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
        return l1_dist


    def connect(self, idx, action=None):
        """
            Connect with another STM when the STM is created by fragmentation.
        """
        loc = self.current.copy() + [self.prev_action]
        if action is not None:
            print(loc, action)
            loc = [loc[0] + action[0], loc[1] + action[1], loc[2]]
        print('connect', loc, self.loc_diff_stms)

        # update distance information
        if len(self.loc_diff_stms) > 0:
            for k, v in self.loc_diff_stms.items():
                dist = [self.distance(loc, l) for l in v]
                min_dist = min(dist)

                if k in self.dist_to_diff_stms:
                    dist = self.dist_to_diff_stms[k].get(idx, 10000000)
                    self.dist_to_diff_stms[k][idx] = min(min_dist, dist)
                else:
                    self.dist_to_diff_stms[k] = {idx: min_dist}

                if idx in self.dist_to_diff_stms:
                    dist = self.dist_to_diff_stms[idx].get(k, 10000000)
                    self.dist_to_diff_stms[idx][k] = min(min_dist, dist)
                else:
                    self.dist_to_diff_stms[idx] = {k: min_dist}


        if idx in self.loc_diff_stms:
            self.loc_diff_stms[idx].append(loc)
        else:
            self.loc_diff_stms[idx] = [loc]


    def update_location(self, dh, dw):
        self.current = [self.current[0]+dh, self.current[1]+dw]
        self.start = [self.start[0]+dh, self.start[1]+dw]
        for k, v in self.loc_diff_stms.items():
            self.loc_diff_stms[k] = [[loc[0]+dh, loc[1]+dw, loc[2]] for loc in v]
        for k, v in self.borders.items():
            v[:,0] += dh
            v[:,1] += dw
            self.borders[k] = v


    def update_discretize(self, pixel_value):
        C, H, W = self.map.shape
        new_map = np.zeros((len(pixel_value), H, W))
        for k, v in self.pixel_value.items():
            new_map[pixel_value[k]] = self.map[v]
        self.map = new_map
        self.pixel_value = pixel_value


    def discretize(self, state):
        """
            Convert the state (3, H, W) to (N, H, W) confidence map. where N is the number of unique colors.
        """
        C, H, W = state.shape
        unique_pixels, discrete_state = np.unique(state.reshape(C, -1).T, return_inverse=True, axis=0)
        N = len(self.pixel_value)
        discrete_state += N
        self.pixel_dtype = unique_pixels.dtype
        for i, pixel in enumerate(unique_pixels):
            key = pixel.tobytes()
            if key in self.pixel_value:
                label = self.pixel_value[key]
                discrete_state[discrete_state==i+N] = label
            else:
                value = len(self.pixel_value)
                self.pixel_value[key] = value
                discrete_state[discrete_state==i+N] = value

        discrete_state = discrete_state.reshape(H,W)
        one_hot_state = self.one_hot_encoding(discrete_state)
        C, H, W = self.map.shape
        new_map = np.zeros((len(one_hot_state)-C, H, W))
        self.map = np.concatenate((self.map, new_map))
        return one_hot_state


    def one_hot_encoding(self, state):
        H, W = state.shape[-2:]
        one_hot_state = np.zeros((max(3, len(self.pixel_value)), H*W))
        one_hot_state[state.reshape(-1), np.arange(H*W)] = 1
        one_hot_state = one_hot_state.reshape(-1, H, W)
        return one_hot_state

    def rotate_state(self, state, action, pos):
        """
            Rotate the egocentric state to allocentric direction.
        """
        if action is None:
            return state, action, pos
        action = self.action_converter[self.prev_action][action]
        deg = 0
        if action == 1:
            deg = 2
            pos = [state.shape[-2]-pos[0]-1, pos[1]]
        elif action == 2:
            deg = 1
            pos = pos[::-1]
        elif action == 3:
            deg = 3
            pos = [pos[1], state.shape[-2]-pos[0]-1]
        state = np.rot90(state, deg, (-2, -1))
        if action < 4:
            self.prev_action = action
        return state, action, pos


    def init_from_gt(self, loc, gt_map, direction):
        if self.discrete:
            gt_map = self.discretize(gt_map)
        self.map = gt_map
        self.observed_step = np.zeros(gt_map.shape[-2:])
        self.current = loc
        self.prev_action = direction


    def update(self, state, action, pos=None):
        """
            Update the local map by using the state
        """

        # find current location in the state 
        if pos is not None:
            pos_in_state = pos
        elif self.fixed_cur_pos is not None:
            pos_in_state = self.fixed_cur_pos
        else:
            raise NotImplementedError
        if self.discrete:
            state = self.discretize(state)

        if self.egocentric:
            state, action, pos_in_state = self.rotate_state(state, action, pos_in_state)
        # Move {0: UP, 1: DOWN, 2: LEFT, 3: RIGHT}
        if action == 0:
            self.current[0] = self.current[0] - 1
        elif action == 1:
            self.current[0] = self.current[0] + 1
        elif action == 2:
            self.current[1] = self.current[1] - 1
        elif action == 3:
            self.current[1] = self.current[1] + 1
        self._update(state, pos_in_state)


    def update_by_rel(self, state, rel_position, ignore_current=True, pos=None):
        """
            Update STM by using relative actions for updating in acting planned actions.
        """
        if self.map.sum() == 0: # first movement
            # initialize location should be empty place
            self.map[0,self.current[0], self.current[1]] = self.init_val
            self.observed_step[self.current[0], self.current[1]] = 1

        if type(state) is torch.Tensor:
            state = state.cpu()
        if pos is None and self.fixed_cur_pos is not None:
            pos = self.fixed_cur_pos

        if self.discrete:
            state = self.discretize(state)

        dh, dw, d = rel_position
        deg = [0, 2, 1, 3][d]
        state = np.rot90(state, deg, (-2, -1))

        if d == 1:
            pos = [state.shape[-2]-pos[0]-1, pos[1]]
        elif d == 2:
            pos = pos[::-1]
        elif d == 3:
            pos = [pos[1], state.shape[-1]-pos[0]-1]

        # current_update
        self.current = [self.current[0] + dh, self.current[1] + dw]
        self.prev_action = d
        self._update(state, pos, ignore_current)



    def _update(self, state, pos_in_state, ignore_current=True):
        self.check_expansions(pos_in_state, state)
        mask_map = np.ones(state.shape)
        if self.discrete and self.decaying_factor > 0:
            self.map[self.map > -1] *= self.decaying_factor
            beta = 1
            alpha = 1-self.decaying_factor 
        else:
            alpha = 1
            beta = 1
        mask_step = np.ones(state.shape[-2:])

        if ignore_current:
            if self.ndim > 2:
                mask_map[:,pos_in_state[0], pos_in_state[1]] = 0
            else:
                mask_map[pos_in_state[0], pos_in_state[1]] = 0
            mask_step[pos_in_state[0], pos_in_state[1]] = 0
        
        up = int(self.current[-2] - pos_in_state[-2])
        down = int(self.current[-2] + state.shape[-2] - pos_in_state[-2]-1) + 1
        left = int(self.current[-1] - pos_in_state[-1])
        right = int(self.current[-1] + state.shape[-1] - pos_in_state[-1]-1) + 1
        if self.ndim > 2:
            self.map[:,up:down, left:right] += self.init_val * state * mask_map # - start from 1 and clamp 1
            self.map[:,up:down, left:right] = self.map[:,up:down, left:right].clip(0, 1)
        else:
            self.map[up:down, left:right] = self.init_val * state * mask_map + self.map[up:down, left:right] * (1-mask_map) * beta
        self.observed_step[up:down, left:right] = self.step * mask_step + self.observed_step[up:down, left:right] * (1-mask_step)
        self.step += 1

        # size limit
        self.check_size()


    def check_size(self):
        """
            Check whether the map should be expanded or not.
        """

        H, W = self.map.shape[-2:]
        h, w = self.current
        mh, mw = self.size_limit
        if h > mh:
            self.map = self.map[:, h-mh:]
            self.observed_step = self.observed_step[h-mh:]
            self.current[0] = mh
        if w > mw:
            self.map = self.map[:, :, w-mw:]
            self.observed_step = self.observed_step[:, w-mw:]
            self.current[1] = mw

        H, W = self.map.shape[-2:]
        h, w = self.current

        if H-h > mh:
            self.map = self.map[:, :h+mh]
            self.observed_step = self.observed_step[:h+mh]
        if W-w > mw:
            self.map = self.map[:, :, :w+mw]
            self.observed_step = self.observed_step[:, :w+mw]


    def get_size(self):
        """
            Calculate the size of the local map.
        """

        known = []
        self.trim()
        for pixel_key, idx in self.pixel_value.items():
            pixel = np.frombuffer(pixel_key, dtype=self.pixel_dtype)
            if np.all(pixel == 0):
                known.append(idx)
            elif not np.all((pixel - self.unknown_marker/255) < 1e-6):
                known.append(idx)

        known_map = self.map[known].sum(0)
        Hs, Ws = np.nonzero(known_map)
        H = Hs.max() - Hs.min() + 1
        W = Ws.max() - Ws.min() + 1

        return len(known), H, W


    def get_map(self, action=None, head=None, dh=0, dw=0, state_shape=(64,5,5)):
        if action is None and head is None:
            self.trim()
            return self.map
        x, y = self.current
        if head is None:
            action = self.action_converter[self.prev_action][action]
            x = x + [-1, 1, 0, 0, 0][action]
            y = y + [0, 0, -1, 1, 0][action]
        else:
            action = head
            x = x + dh
            y = y + dw
        C, H, W = self.map.shape
        c, h, w = state_shape
        if action == 4:
            action = self.prev_action

        if action >= 2: # swap (h,w)
            state = np.zeros((c, state_shape[-1], state_shape[-2]))
        else:
            state = np.zeros((state_shape))

        if action == 0:
            x1 = max(h-1-x, 0)
            x2 = h
            y1 = max(w//2-y, 0)
            y2 = w//2 + min(w-(w//2), W-y)
            state[:C, x1:x2, y1:y2] = self.map[:, max(x-h+1,0):x+1, max(y-(w//2),0):y+w-(w//2)]
        elif action == 1:
            x1 = 0
            x2 = min(H-x, h)
            y1 = max(w//2-y, 0)
            y2 = w//2 + min(w-(w//2), W-y)
            state[:C, x1:x2, y1:y2] = self.map[:, x:x+h, max(y-(w//2),0):y+w-(w//2)]
            state = np.rot90(state, 2, (-2, -1))
        elif action == 2:
            x1 = max(w//2-x, 0)
            x2 = w//2 + min(w-(w//2), H-x)
            y1 = max(h-1-y, 0)
            y2 = h
            state[:C, x1:x2, y1:y2] = self.map[:, max(x-(w//2),0):x+w-(w//2), max(y-h+1, 0):y+1]
            state = np.rot90(state, 3, (-2, -1))
        elif action == 3:
            x1 = max(w//2-x, 0)
            x2 = w//2 + min(w-(w//2), H-x)
            y1 = 0
            y2 = min(h, W-y)
            state[:C, x1:x2, y1:y2] = self.map[:, max(x-(w//2),0):x+w-(w//2), y:y+h]
            state = np.rot90(state, 1, (-2, -1))
        return state


    def rotate(self, degree):
        """
            Rotate the entire map.
        """
        H, W = self.map.shape[-2:]
        self.map = np.rot90(self.map, degree, axes=(-2, -1))

        if degree == 2: # action 1
            self.current = [H-self.current[0], W-self.current[1]]
            self.start = [H-self.start[0], W-self.start[1]]
            for k, v in self.loc_diff_stms.items():
                self.loc_diff_stms[k] = [[H-loc[0], W-loc[1]] for loc in v]
        elif degree == 1: # leg
            self.current = [W-self.current[1], self.current[0]]
            self.start = [W-self.start[1], self.start[0]]
            for k, v in self.loc_diff_stms.items():
                self.loc_diff_stms[k] = [[W-loc[1], loc[0]] for loc in v]
        elif degree == 3: # leg
            self.current = [self.current[1], H-self.current[0]]
            self.start = [self.start[1], H-self.start[0]]
            for k, v in self.loc_diff_stms.items():
                self.loc_diff_stms[k] = [[loc[1], H-loc[0]] for loc in v]


    def get_step(self):
        self.trim()
        return self.observed_step


    def check_expansions(self, pos_in_state, state):
        cond = True
        while cond:
            d_up = pos_in_state[-2]
            d_down = int(state.shape[-2] - pos_in_state[-2])
            d_left = pos_in_state[-1]
            d_right = int(state.shape[-1] - pos_in_state[-1])
            if self.current[-2] - d_up < 0:
                self.expansion(0)
            elif self.current[-2] + d_down > self.map.shape[-2]:
                self.expansion(1)
            elif self.current[-1] - d_left < 0:
                self.expansion(2)
            elif self.current[-1] + d_right > self.map.shape[-1]:
                self.expansion(3)
            else:
                cond = False

    def get_empty(self):
        """
            Get N_empty.
        """
        empty = []
        wall = []

        for pixel_key, idx in self.pixel_value.items():
            pixel = np.frombuffer(pixel_key, dtype=self.pixel_dtype)
            if np.all(pixel == 0):
                empty.append(idx)
            elif not np.all((pixel - self.unknown_marker/255) < 1e-6):
                wall.append(idx)
        empty = self.map[empty]
        wall = self.map[wall]
        nonzero_mask = (empty.sum(0) > 0).astype(int)
        empty = np.concatenate((empty, wall)).argmax(0) == 0
        empty = empty & nonzero_mask
        return empty

    def get_occupancy(self):
        """
            Get N_occupied.
        """
        empty = []
        unknown = []
        wall = []

        N = len(self.map)
        h_line = np.zeros((N, 1, self.map.shape[-1]))
        v_line = np.zeros((N, self.map.shape[-2] + 2, 1))
        padded_map = np.concatenate((h_line, self.map, h_line), axis=-2)
        padded_map = np.concatenate((v_line, padded_map, v_line), axis=-1)

        for pixel_key, idx in self.pixel_value.items():
            pixel = np.frombuffer(pixel_key, dtype=self.pixel_dtype)
            if np.all(pixel == 0):
                empty.append(idx)
            elif np.all(pixel == self.unknown_marker/255):
                unknown.append(idx)
            else:
                wall.append(idx)

        empty = padded_map[empty].sum(0) > 0 
        wall = padded_map[wall].sum(0) > 0
        occupied =  empty + wall > 0
        rows, cols = np.nonzero(np.logical_not(occupied))
        H, W = occupied.shape
        unknown = np.zeros((H, W))
        for h, w in zip(rows, cols):
            flag = False
            for dh in [-1, 0, 1]:
                if flag:
                    break
                for dw in [-1, 0, 1]:
                    if h + dh < 0 or h + dh == H or w + dw < 0 or w + dw == W:
                        continue
                    if empty[h+dh, w+dw]:
                        unknown[h, w] = 1
                        flag = True
                        break
        return unknown, empty, wall

    def _refine_subgoals(self, subgoals, empty, unknown):
        locations = np.stack(np.nonzero(empty )).T.reshape(1, -1, 2)
        subgoals = subgoals.reshape(-1, 1, 2)
        # broadcasting (N, 2, 1) - (1, M, 2) => (N, M, 2)
        diff = subgoals - locations
        distances = np.linalg.norm(diff, axis=-1)
        indices = distances.argmin(axis=-1)
        subgoals = locations[0][indices]
        return subgoals


    def frontier_goals(self, mode='average'):
        """
            Find subgoal for Frontier-based Exploration.
        """

        unknown, empty, wall = self.get_occupancy()
        # clustering
        clustered = unknown - 1
        if unknown.sum() == 0:
            return None
        rows, cols = np.nonzero(unknown)
        idx = 1
        sizes = []
        size_th = 1
        subgoals = []
        unknown_coor = []
        ch, cw = self.current
        for row, col in zip(rows, cols):
            if clustered[row, col] == 0:
                clustered = masking(clustered, row, col, idx, ignore_current=False)
                frontier_map = clustered == idx

                size = frontier_map.sum()
                sizes.append(size)
                hs, ws = np.nonzero(frontier_map)
                unknown_coor.append((hs, ws))
                if mode == 'average':
                    h = hs.mean()
                    w = ws.mean()
                elif mode == 'farthest':
                    dist = abs(hs - ch) + abs(ws - cw)
                    idx = dist.argmax()
                    h = hs[idx]
                    w = ws[idx]
                elif mode == 'nearest':
                    dist = abs(hs - ch) + abs(ws - cw)
                    idx = dist.argmin()
                    h = hs[idx]
                    w = ws[idx]
                elif mode == 'nearest_centroid':
                    h = hs.mean()
                    w = ws.mean()
                    dist = abs(hs - h) + abs(ws - w)
                    idx = dist.argmin()
                    h = hs[idx]
                    w = ws[idx]
                elif mode == 'farthest_centroid':
                    h = hs.mean()
                    w = ws.mean()
                    dist = abs(hs - h) + abs(ws - w)
                    idx = dist.argmax()
                    h = hs[idx]
                    w = ws[idx]
                    
                subgoals.append([h,w])
                idx += 1
        
        subgoals = np.asarray(subgoals)
        sizes = np.asarray(sizes)
        idx = sizes > size_th
        if idx.sum() == 0:
            idx = sizes == sizes.argmax()
        subgoals = subgoals[idx]
        sizes = sizes[idx]
        subgoals = self._refine_subgoals(subgoals, empty, unknown)

        # evade frontier location as wall
        # remove padding
        subgoals -= 1
        # change to relative position
        subgoals[:,0] = subgoals[:,0] - self.current[0]
        subgoals[:,1] = subgoals[:,1] - self.current[1]
        distance = np.linalg.norm(subgoals, axis=-1)
        order = distance.argsort()

        # head direction
        heads = np.zeros((len(subgoals), 1), dtype=int)
        for i, subgoal in enumerate(subgoals):
            h, w = subgoal
            if -h > w:
                if -h > -w:
                    d = 0
                else:
                    d = 2
            else:
                if -h > -w:
                    d = 3
                else:
                    d = 1
            heads[i] = d
        subgoals = np.concatenate((subgoals, heads), axis=-1)

        subgoals = subgoals[order]
        distance = distance[order]
        sizes = sizes[order]
        empty = empty[1:-1, 1:-1]
        unknown = unknown[1:-1, 1:-1]

        return subgoals, distance, sizes, (empty, unknown)

    def trim(self):
        """
        Trim the map
        """
        row = self.observed_step.max(axis=-1)
        row = np.nonzero(row > -1)[0]
        col = self.observed_step.max(axis=-2)
        col = np.nonzero(col > -1)[0]
        if len(row) == 0 or len(col) == 0:
            return
        if len(self.map.shape) > 2:
            self.map = self.map[:,row[0]:row[-1]+1, col[0]:col[-1]+1]
        else:	
            self.map = self.map[row[0]:row[-1]+1, col[0]:col[-1]+1]
        self.observed_step = self.observed_step[row[0]:row[-1]+1, col[0]:col[-1]+1]
        self.update_location(-row[0], -col[0])
        

    def forgetting(self, threshold):
        self.map[self.map < threshold] = 0
        self.trim()
        return self.map


    def expansion(self, option=-1):
        """
            Increase the size of the map.
        """
    #		print('expansion', option)
        tile_shape = list(self.map.shape)
        if option <= 1: # UP/DOWN
            tile_shape[-2] = self.expansion_unit[-2]
            tile = -np.ones(tile_shape) + (1 if self.discrete else 0)
            tile_step = - np.ones(tile_shape[-2:])
            if option == 0: # UP
                self.map = np.concatenate((tile, self.map), axis=-2)
                self.observed_step = np.concatenate((tile_step, self.observed_step), axis=-2)
                self.update_location(tile_shape[-2], 0)
            else: # DOWN
                self.map = np.concatenate((self.map, tile), axis=-2)
                self.observed_step = np.concatenate((self.observed_step, tile_step))
#            self.expansion_unit[-2] = min(200, int(self.expansion_unit[-2] + 15)) # update this size of tile
        elif option >= 2: # LEFT/RIGHT
            tile_shape[-1] = self.expansion_unit[-1]
            tile = -np.ones(tile_shape) + (1 if self.discrete else 0)
            tile_step = - np.ones(tile_shape[-2:])
            if option == 2: # LEFT
                self.map = np.concatenate((tile, self.map), axis=-1)
                self.observed_step = np.concatenate((tile_step, self.observed_step), axis=-1)
                self.update_location(0, tile_shape[-1])
            else: # RIGHT
                self.map = np.concatenate((self.map, tile), axis=-1)
                self.observed_step = np.concatenate((self.observed_step, tile_step), axis=-1)
#            self.expansion_unit[-1] = min(200, int(self.expansion_unit[-1] + 15)) # update this size of tile
                

if __name__ == '__main__':
    memory = MappingMemory(shape=(3,5,5), fixed_cur_pos=(4,2), decaying_factor=0.7, egocentric=True, discrete=True)
    state = np.arange(25).reshape(5,5)
    state = np.stack([state, state, state])
    import copy
    memory2 = copy.deepcopy(memory)
    memory2.update_by_rel(state, (-3,-1, 2), pos=(0,0))
    memory.update_by_rel(state, (-3,-1, 2))
    print(memory.map.argmax(0), memory.current)
    print(memory2.map.argmax(0), memory2.current)
    memory.update_by_rel(state, (-3,-1, 0))
    memory2.update_by_rel(state, (-3,-1, 0), pos=(0,0))
    print(memory.map.argmax(0), memory.current)
    print(memory2.map.argmax(0), memory2.current)
    breakpoint()
    if False:
        # discretize test
        states = np.zeros((3,2,2))
        states[:,0,0] = 1
        states[1,1,1] = 2
        print(states)
        print(memory.discretize(states))
        states = np.zeros((3,2,2))
        states[:,0,0] = 2
        states[1,1,1] = 3
        print(states)
        print(memory.discretize(states))

    for action in [None, 0, 4, 2, 3, 0, 3, 1, 1, 1]:
        memory.update(state, action)
        memory.get_map(action)
        print(memory.get_map()[0], action)
        memory.frontier_goals()
    memory.get_step()
