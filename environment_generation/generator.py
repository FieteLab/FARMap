# Map Generator written by Authors.


import os
import random
import cv2

import numpy as np


def get_start_loc(curr_loc, env, verbose=True):
    if curr_loc is None or len(curr_loc[0]) == 0:
        # create starting position
        rows, cols = (env.sum(0)==0).nonzero()
        idx = np.random.randint(len(rows))
        h, w = rows[idx], cols[idx]
        d = np.random.randint(4)
        if verbose:
            print("Start from random location: (X: {}, Y: {}, D: {})".format(h, w, d))
    else: 
        h, w = curr_loc
        d = 0

    return h, w, d


def get_map(maze_id, num_envs=1, scale=3, generation=False, maze_path='dataset.pkl'):
    mazes = []
    for _ in range(num_envs):
        with open(maze_path, 'rb') as f:
            maze = pickle.load(f)[maze_id]
        colorize_maze(maze)
        maze, _ = enlarge_map(maze, (5,5), scale=scale)
        mazes.append(maze)
    maze = np.concatenate(mazes, 0)

    # randomly initialize the starting position
    curr_loc = get_start_loc(None, maze)
    return maze, curr_loc



def enlarge_map(maze, window, scale=1): # scale up the maze
    window = (int(window[0]*scale), int(window[1]*scale))
    h, w = (maze[0] == 255).nonzero()
    use_start_pos = len(h) > 0
    if use_start_pos:
        maze[:, h,w] = 0
        h, w = int(h*scale), int(w*scale)
        h += scale-1
        w += scale//2
    C, H, W = maze.shape
    maze = maze.transpose(1,2,0)
    maze = cv2.resize(maze, (scale*W, scale*H), interpolation=cv2.INTER_NEAREST)
    maze = maze.transpose(2, 0, 1)
    if use_start_pos:
        maze[:, h,w] = 255
    if window[1] % 2 == 0:
        window = (window[0], window[1] + 1)

    return maze, window


def get_color_map(n=256):
    color_map = np.zeros((n, 3))
    for i in range(n):
        r = b = g = 0
        cid = i
        for j in range(0, 8):
            r = np.bitwise_or(r, np.left_shift(np.unpackbits(np.array([cid], dtype=np.uint8))[-1], 7-j))
            g = np.bitwise_or(g, np.left_shift(np.unpackbits(np.array([cid], dtype=np.uint8))[-2], 7-j))
            b = np.bitwise_or(b, np.left_shift(np.unpackbits(np.array([cid], dtype=np.uint8))[-3], 7-j))
            cid = np.right_shift(cid, 3)
        color_map[i][0] = r
        color_map[i][1] = g
        color_map[i][2] = b
    return color_map


def colorize_maze(maze, target='%'):
    target_map = maze[0] == ord(target)
    N = target_map.sum()
    colors = get_color_map(N+1)[1:]
    np.random.shuffle(colors)
    rows, cols = target_map.nonzero()
    for row, col, c in zip(rows, cols, colors):
        maze[:, row, col] = c
    return maze




def iterative_masking(env, h, w, idx):
    if env[h, w] != 0 :
        return env
    queue = [[h, w]]
    H, W = env.shape
    while len(queue) > 0:
        h, w = queue[0]
        queue = queue[1:]
        if env[h, w] != 0:
            continue

        env[h, w] = idx
        for dh, dw in zip([1, -1, 0, 0], [0, 0, -1, 1]):
            if h + dh < 0 or h + dh == H or w + dw < 0 or w + dw == W:
                continue
            if env[h+dh, w+dw] == 0:
                queue.append([h+dh, w+dw])
    return env


class MapGenerator():
    def __init__(self, room_size, map_size, corridor_info={}, num_flip=0, min_size=25, visualize=False):
        self.room_size = room_size
        self.map_size = map_size
        self.num_flip = num_flip
        self.min_size = min_size
        self.visualize = visualize
        self.save_path = 'vis.png'

        self.corridor_length = corridor_info.get('length', 2)
        st_v = corridor_info.get('st_v', 0)
        st_h = corridor_info.get('st_h', 0)
        width_v = corridor_info.get('width_v', 1)
        width_h = corridor_info.get('width_h', 1)
        width_v = min(room_size[0], width_v)
        width_h = min(room_size[1], width_h)
        self.v_loc = np.arange(st_v, st_v+width_v)
        self.h_loc = np.arange(st_h, st_h+width_h)
        print(self.__dict__)

    def divide_isolated_map(self, env):
        idx = 1
        maps = []
        while True:
            rows, cols = (env==0).nonzero()
            if len(rows) == 0:
                break
            env = iterative_masking(env, rows[0], cols[0], idx)
            candidate = (env == idx).astype(int)
            if candidate.sum() > self.min_size:
                maps.append(1-candidate)
            idx += 1
        if len(maps) == 0:
            return None
        maps = np.stack(maps)
        for m in maps:
            print(m.size-m.sum())
            print(m)

        return maps


    def init_env(self):
        H = (self.room_size[0] + self.corridor_length) * self.map_size[0] + 1
        W = (self.room_size[1] + self.corridor_length) * self.map_size[1] + 1
        env = np.zeros((H,W), dtype=int)

        # horizontal wall
        for i in range(self.map_size[0]+1):
            st = i*(self.corridor_length+self.room_size[0])
            env[st:st+self.corridor_length] = -1

        for i in range(self.map_size[1]+1):
            st = i*(self.corridor_length+self.room_size[1])
            env[:,st:st+self.corridor_length] = -1
        return env


    def connect_rooms(self, env, v_connections, h_connections, v_loc=0, h_loc=0):
        rows, cols = v_connections.nonzero()
        for row, col in zip(rows, cols):
            i = (1+row) * (self.room_size[0] + self.corridor_length)
            j = col * (self.room_size[1] + self.corridor_length) + self.corridor_length
            try:
                env[i:i+self.corridor_length, j+h_loc] = 0
            except:
                breakpoint()
        rows, cols = h_connections.nonzero()
        for row, col in zip(rows, cols):
            i = (1+col) * (self.room_size[1] + self.corridor_length)
            j = row * (self.room_size[0]+self.corridor_length) + self.corridor_length
            env[j+v_loc, i:i+self.corridor_length] = 0
        return env

    def merge_rooms(self, env, v_connections, h_connections):
        return self.connect_rooms(env, v_connections, h_connections, np.arange(self.room_size[0]), np.arange(self.room_size[1]))


    def flip_boundaries(self, env, p_flip, sign=-1):
        walls = env != 0
        coordinates = []
        H, W = env.shape
        for h in range(H):
            for w in range(W):
                v = walls[h,w]
                for dh, dw in zip([1, -1, 0, 0], [0, 0, -1, 1]):
                    if h + dh < 0 or h + dh == H or w + dw < 0 or w + dw == W:
                        continue
                    if int(v)^int(walls[h+dh, w+dw]) == 1:
                        coordinates.append((h,w))
                        break

        coordinates = np.asarray(coordinates)
        flip = np.random.random(len(coordinates)) < p_flip

        for (h, w), f in zip(coordinates, flip):
            env[h,w] = sign * (int(walls[h,w]) ^ f)

        return env


    def generate_env(self, p_connect=0, p_merge=0, p_flip=0, n=0):
        env = self.init_env()
        if self.visualize:
            visualize_map(-env[np.newaxis], f'dataset/{n}_init.png')

        v_connections = np.random.random((self.map_size[0]-1, self.map_size[1])) < p_connect
        h_connections = np.random.random((self.map_size[0], self.map_size[1]-1)) < p_connect
        env = self.connect_rooms(env, v_connections, h_connections, self.v_loc, self.h_loc)
        if self.visualize:
            visualize_map(-env[np.newaxis], f'dataset/{n}_connect.png')

        v_connections = np.random.random((self.map_size[0]-1, self.map_size[1])) < p_merge
        h_connections = np.random.random((self.map_size[0], self.map_size[1]-1)) < p_merge
        env = self.merge_rooms(env, v_connections, h_connections)
        if self.visualize:
            visualize_map(-env[np.newaxis], f'dataset/{n}_merge.png')


        for i in range(self.num_flip):
            env = self.flip_boundaries(env, p_flip)
            if self.visualize:
                visualize_map(-env[np.newaxis], 'dataset/{}_flip_{}.png'.format(n,i))

        envs = self.divide_isolated_map(env)

        if self.visualize:
            visualize_map(envs, self.save_path)

        envs = [crop(env) for env in envs]
        print("ENVIRONMENTS")
        for env in envs:
            print(env)

        return envs

def crop(env):
    empty = env == 0
    cropped = env[empty.sum(1)>0][:,empty.sum(0)>0]
    padding = np.ones((cropped.shape[0],1))
    cropped = np.concatenate((padding, cropped, padding), axis=1)
    padding = np.ones((1,cropped.shape[1]))
    cropped = np.concatenate((padding, cropped, padding), axis=0)
    return cropped


def convert2ascii_map(env, wall='%', num_channel=3):
    env *= ord(wall)
    env = env.astype(np.uint8)
    env = np.stack([env for _ in range(num_channel)])
    return env

def visualize_map(envs, path):
    L, H, W = envs.shape
    N = int(np.ceil(L**0.5))
    M = int(np.ceil(L/N))
    vis_map = []
    for n in range(N):
        vis_map_chunk = []
        for m in range(M):
            i = n * M + m
            if i < L:
                vis_map_chunk.append(envs[i])
            else:
                vis_map_chunk.append(np.zeros(envs[0].shape))
            vis_map_chunk.append(np.zeros((H, 1)) + 0.5)
        vis_map_chunk = np.concatenate(vis_map_chunk, axis=1)
        vis_map_chunk = np.concatenate((vis_map_chunk, 0.5+np.zeros((1,vis_map_chunk.shape[1]))), axis=0)
        vis_map.append(vis_map_chunk)
    vis_map = np.concatenate(vis_map, 0)

    scale = int(800/H)
    vis_map = (255*vis_map).astype(np.uint8)
    if scale > 1:
        vis_map = cv2.resize(vis_map, (scale*W, scale*H), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, vis_map)

if __name__ == '__main__':
    seed = 13
    import pickle
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    N = 200
    total_envs = []

    attrs = []
    num_envs = []
    for n in range(N):
        size = random.randint(3,7)
        room = random.randint(3,7)

        SIZE = (size,size) # room layout
        room_size = (room,room) # room size
        num_flip = random.randint(0,10) # the number of flip boundary regions
        min_size = (room **2) * 3 #25 # minimal map size
        visualize = False
        corridor_info = {
                'length': random.randint(1,3), # length of corridor
                'st_v': 0, # starting index of vertical corridor connection, [0, room_size[0])
                'width_v': random.randint(1, room-1), # width of vertical corridor
                'st_h': 0, # starting index of horizontal corridor connection [0, room_size[1])
                'width_h': random.randint(1, room-1), # width of horizontal corridor
                }

        p_connect = 0.25 # probability of connecting two rooms
        p_merge = 0.25 # probability of merging two rooms
        p_flip = 0.05 # probability of flipping boundaries

        generator = MapGenerator(room_size, SIZE, corridor_info, num_flip, min_size, visualize=visualize)
        envs = generator.generate_env(p_connect, p_merge, p_flip, len(total_envs))
        total_envs = total_envs + envs
        attr = generator.__dict__
        attrs.append(attr)
        num_envs.append(len(envs))
    
    print(len(total_envs))

    # shuffle
    order = np.random.permutation(np.arange(len(total_envs)))
    envs = []
    for i, idx in enumerate(order):
        print(i, idx)
        envs.append(total_envs[idx])

    if not os.path.exists('dataset'):
        os.mkdir('dataset')
    else:
        os.system('rm -rf dataset/*')
    for i, env in enumerate(envs):
        path = 'dataset/{:03}.png'.format(i)
        visualize_map(env[np.newaxis,:,:], path)
        print(env.shape)

    ascii_envs = [convert2ascii_map(env) for env in envs]


    with open('gen_env.pkl', 'wb') as f:
        pickle.dump(ascii_envs, f)
    with open('config.pkl', 'wb') as f:
        pickle.dump([attrs, num_envs, order], f)
