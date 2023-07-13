import os
import pickle
import cv2
import tqdm
import random
import numpy as np

import torch
from torch import nn
from torch.distributions.normal import Normal
import torch.backends.cudnn as cudnn


import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm


def one_hot_encoding(state, C):
    H, W = state.shape[-2:]
    one_hot_state = np.zeros((C, H*W))
    one_hot_state[state.reshape(-1), np.arange(H*W)] = 1
    one_hot_state = one_hot_state.reshape(-1, H, W)
    return one_hot_state

def check_discard(pixel, unknown=ord('X')):
    for p in pixel:
        if abs(p - -1/255) > 1e-6 and abs(p - unknown /255) > 1e-6:
            return False
    return True


def get_score(pred, gt, pixel_value, unknown=ord('X')):
    """
        Return Confidence Score
    """

    C, H, W = gt.shape
    unique_pixels, discrete_gt = np.unique(gt.reshape(C, -1).T, return_inverse=True, axis=0)
    discrete_gt = discrete_gt.reshape(H, W)
    one_hot = one_hot_encoding(discrete_gt, len(unique_pixels))
    discard = 0 # current location
    for i, pixel in enumerate(unique_pixels):
        if check_discard(pixel, unknown=unknown):
            discard += one_hot[i].sum()
            one_hot[i] = 0
            continue
        key = pixel.tobytes()
        if key in pixel_value:
            c = pixel_value[key]
            one_hot[i] *= pred[c]
        else:
            one_hot[i] *= 0
    
    prob = one_hot.sum(0)
    if prob.size > discard:
        score = prob.sum() / (prob.size-discard)
    else:
        score = 0
    return score 




def masking(env, h, w, idx, mode='square', ignore_current=True):
    if env[h, w] != 0:
        return env
    H, W = env.shape
    queue = [[h,w]]
    step = 0
    stored_env = env.copy()
    max_iter = (env==0).sum()
    visited = {}
    while len(queue) > 0:
        step += 1
        h,w = queue[0]
        visited[(h,w)] = 1
        env[h, w] = idx
        queue = queue[1:]
        flag = False
        if mode == 'cross': # criteria for cell conectivity
            for dh, dw in zip([-1, +1, 0, 0], [0, 0, -1, +1]):
                if h + dh < 0 or h + dh == H or w + dw < 0 or w + dw == W:
                    continue
                if env[h+dh, w+dw] == 0:
                    queue.append([h+dh, w+dw])
                    break

        if mode == 'square':
            for dh in [-1, 0, 1]:
                for dw in [-1, 0, 1]:
                    if dh == 0 and dw == 0:
                        continue
                    if h + dh < 0 or h + dh == H or w + dw < 0 or w + dw == W:
                        continue
                    if env[h+dh, w+dw] == 0 and (h+dw, w+dw) not in visited:
                        queue.append([h+dh, w+dw])
                    if env[h+dh, w+dw] == -1:
                        env[h+dh, w+dw] = 100
    return env


def discovery(h, w, d, obs, observed_map, env, padding=15):
    observed = (obs == ord('X')).sum(0) != 3
    H, W = obs.shape[-2:]
    P = padding
    dh1s = [-H+1, +0, -(W//2), -(W//2)]
    dh2s = [+1, H, W-(W//2), W-(W//2)]
    dw1s = [-(W//2), -(W//2), -H+1, +0]
    dw2s = [W-(W//2), W-(W//2), +1, H]

    rotation_factor = [0, 2, 1, 3]
    dh1, dh2 = dh1s[d], dh2s[d]
    dw1, dw2 = dw1s[d], dw2s[d]
    degree =rotation_factor[d]
    observed = np.rot90(observed, degree)
    observed_map[P+h+dh1:P+h+dh2, P+w+dw1:P+w+dw2][observed] += 1
    return observed_map


def set_seed(seed=13, verbose=True):
    if verbose:
        print("Set Seed:", seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def size_visible_area(env, mode='cross'):
    mask = env.sum(0)
    H, W = mask.shape
    count = 0
    for h in range(H):
        for w in range(W):
            if mask[h, w] == 0:
                count += 1
                continue
            if mode == 'square':
                flag = False
                for dh in [-1, 0 , 1]:
                    if flag:
                        break
                    for dw in [-1, 0, 1]:
                        if h + dh < 0 or h + dh == H or w + dw < 0 or w + dw == W:
                            continue
                        if mask[h+dh, w+dw] == 0:
                            count += 1
                            flag = True
                            break
            elif mode == 'cross':
                for dh, dw in zip([-1, +1, 0, 0], [0, 0, -1, +1]):
                    if h + dh < 0 or h + dh == H or w + dw < 0 or w + dw == W:
                        continue
                    if mask[h+dh, w+dw] == 0:
                        count += 1
                        break

    return count



def get_map(maze_id, num_envs=1, scale=3, generation=False, maze_path='gen_env.pkl', obs_size=(5,5), wall=ord('%'), empty=0):
    mazes = []
    original_mazes = []
    for _ in range(num_envs):
        if maze_id == -1: # specialized large empty room for testing
            maze = np.zeros((3, 200, 200)) + wall
            maze[:, 1:-1, 1:-1] = empty
        elif generation:
            with open(maze_path, 'rb') as f:
                maze = pickle.load(f)[maze_id]
        else:
            maze = convertMAZE2map()[maze_id]
        original_mazes.append(np.copy(maze))
        colorize_maze(maze, target=wall)
        maze, _ = enlarge_map(maze, obs_size, scale=scale)
        row, col = ((maze==255).sum(0)==3).nonzero() # agent start location in pycolab.
        maze[:,row,col] = empty
        mazes.append(maze)

    maze = np.concatenate(mazes, 0)
    original_maze = np.concatenate(original_mazes, 0)
    return maze, (row, col), original_maze

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


def get_avg_surprisal(trajs, scores, maze, save_dir=''):
    if trajs.shape[1] == 6:
        trajs = trajs[:, 3:]
    else:
        trajs = trajs[1:]
    vis_set, inverse = np.unique(trajs, axis=0, return_inverse=True)
    avg_scores = np.stack([scores[inverse==i].mean() for i in range(len(vis_set))])
    
    score_map = np.zeros((4, maze.shape[-2], maze.shape[-1]))
    used_map = np.zeros((4, maze.shape[-2], maze.shape[-1]))
    step = 0
    for (h, w, d), s in zip(vis_set, avg_scores):
        score_map[d, h, w] = 1-s # confidence to surprisal
        used_map[d, h, w] = 1
        step += 1

    return score_map, used_map

def merge_surprisal(observations, locations, predictions, env_ids, mazes, visualize=True, save_path=None):
    """
        Used for Visualization
    """
    # locations: set of (h, w, d)
    # predictions: set of (4, H_o, W_o)
    env_ids = torch.cat(env_ids).numpy()
    if len(locations[0].shape) == 2:
        locations = torch.cat(locations)
        observations = torch.cat(observations)
        predictions = torch.cat(predictions)
    vis_maze = None
    Ho, Wo = observations[0].shape[-2:]


    P = max(Ho, Wo) # padding
    dh1s = [-Ho+1, +0, -(Wo//2), -(Wo//2)]
    dh2s = [+1, Ho, Wo-(Wo//2), Wo-(Wo//2)]
    dw1s = [-(Wo//2), -(Wo//2), -Ho+1, +0]
    dw2s = [Wo-(Wo//2), Wo-(Wo//2), +1, Ho]
    degrees = [0, 2, 1, 3]
    

    unique = np.unique(env_ids)
    vis_mazes = []
    surprisal_maps = []
    if type(mazes) is not list:
        H, W = mazes.shape[1:]
        mazes = mazes.reshape(-1, 3, H, W)

    for env_id in unique:
        env_obs = observations[env_ids == env_id]
        env_locs = locations[env_ids == env_id]
        env_preds = predictions[env_ids == env_id]

        maze = mazes[int(env_id)]
        H, W = maze.shape[1:]
        surprisal_map = torch.zeros((4, H + 2*P, W + 2*P))
        count = torch.zeros((1, H + 2*P, W + 2*P))
        for obs, loc, pred in zip(env_obs, env_locs, env_preds):
            if len(obs.shape) == 4:
                obs = obs[0]
            h, w, d = loc
            d = int(d)
            degree = degrees[d]
            pred = torch.rot90(pred, degree, dims=(-2,-1))
            obs = torch.rot90(obs, degree, dims=(-2,-1))
            h1 = max(h + dh1s[d] + P, 0)
            h2 = min(h + dh2s[d] + P, H + 2*P)
            w1 = max(w + dw1s[d] + P, 0)
            w2 = min(w + dw2s[d] + P, W + 2*P)
            mask = (obs.sum(0, keepdim=True) == 0)
            scores = pred * mask

            surprisal_map[:, h1:h2, w1:w2] += scores
            count[:, h1:h2, w1:w2] += mask

        # remove padding
        surprisal_map = surprisal_map[:,P:-P][:,:,P:-P]
        count = count[:,P:-P][:,:,P:-P]

        mask = np.logical_or((maze.sum(0) == 0), (maze==255).sum(0)==3)
        surprisal_map[:, mask] = surprisal_map[:,mask] / count[:,mask]
        
        if visualize:
            surprisal_map = surprisal_map.numpy()
            vis_maze1 = np.concatenate(surprisal_map[:2], axis=-1)
            vis_maze2 = np.concatenate(surprisal_map[2:], axis=-1)
            vis_maze = np.concatenate((vis_maze1, vis_maze2), axis=-2)

            mask = np.concatenate((mask, mask), axis=-1)
            mask = np.concatenate((mask, mask), axis=-2)

            copy_map = maze.transpose(1,2,0)
            copy_map = np.concatenate((copy_map, copy_map), axis=0)
            copy_map = np.concatenate((copy_map, copy_map), axis=1)

            vis_maze *= 255
            vis_maze = vis_maze[:,:,np.newaxis]
            vis_maze = vis_maze.astype(np.uint8)
            vis_maze = cv2.applyColorMap(vis_maze, cv2.COLORMAP_JET)
            vis_maze[mask==0] = copy_map[mask==0]
            scale = max(1, int(800/H))
            vis_maze = cv2.resize(vis_maze, (scale*W, scale*H), interpolation=cv2.INTER_NEAREST)
            if save_path is not None:
                cv2.imwrite(save_path.replace('.png', '_{}.png'.format(int(env_id))), vis_maze)
        surprisal_maps.append(surprisal_map)
        vis_mazes.append(vis_maze)

    return surprisal_maps, vis_mazes


def enlarge_map(maze, window, scale=1, use_start_pos=True):
    window = (int(window[0]*scale), int(window[1]*scale))
    h, w = (maze[0] == 255).nonzero()
    use_start_pos = len(h) > 0 and use_start_pos
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
    """
        Return color map following the colorization code of PASCAL VOC 2007 dataset.
    """
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


def colorize_maze(maze, target=ord('%'), colors=None):
    target_map = maze[0] == target
    N = target_map.sum()
    if colors is None:
        colors = get_color_map(N+1)[1:]
    np.random.shuffle(colors)
    maze[:, target_map] = colors.T
#    rows, cols = target_map.nonzero()
#    for row, col, c in zip(rows, cols, colors):
#        maze[:, row, col] = c
    return maze


def remap_checker(scores, threshold, lower=None, T=1, remap_loc=[], dist=True, rho=2.0, mode='z'):
    """
        Check whether the current position is fraction point or not.
        If the current position is fraction point, return True.
        Otherwise, return False.
    """
    if mode == 'random':
        return np.random.uniform() <= rho

    elif mode == 'uniform':
        if len(remap_loc) > 0:
            scores = scores[remap_loc[-1]:]
        return len(scores) >= rho

    if len(scores) < 2*T:
        return False
    if dist:
        if len(remap_loc) > 0:
            scores = scores[remap_loc[-1]:]
        
        sample = scores[-1]
        scores = scores[:-1]
        if len(scores) < 26: # sample size is too small
            return False
            
        mean, std = np.mean(scores), np.std(scores, ddof=1) # sample std
        if mode == 'z':
            target = -(sample - mean) / std
        elif mode == 'ratio': # sample and mean is confidence values.
            target = (1-sample) / (1-mean + 1e-9)
        else:
            raise Exception("Wrong Mode")
        return  target > rho

    if lower is None:
        lower = threshold

    neg = scores[-2*T:-T]
    for n in scores[-2*T:-T]:
        if n >= lower:
            return False
    for p in scores[-T:]:
        if p < threshold:
            return False
    return True



def convert_conf_map(confidence_maps, currents, env):
    """
        place confidence map into the exact location of the environment for visualization.
    """
    if type(confidence_maps) is not list:
        confidence_maps = [confidence_maps]
        breakpoint()

    conf_maps = []
    H, W = env.shape[1:]
    P = 15
    for (c_map, loc), curr in zip(confidence_maps, currents):
        Hc, Wc = c_map.shape
        conf_map = np.zeros((H, W))
        h1 = curr[0] - loc[0]
        h2 = curr[0] + Hc - loc[0]
        w1 = curr[1] - loc[1]
        w2 = curr[1] + Wc - loc[1]
        hh1 = 0
        hh2 = Hc
        ww1 = 0
        ww2 = Wc
        if h1 < 0:
            hh1 = -h1
            h1 = 0
        if h2 > H:
            hh2 = H - h2
            h2 = H
        if w1 < 0:
            ww1 = -w1
            w1 = 0
        if w2 > W:
            ww2 = W - w2
            w2 = W
        try:
            conf_map[h1:h2, w1:w2] = c_map[hh1:hh2, ww1:ww2]
        except:
            breakpoint()
        conf_maps.append(conf_map)

    return conf_maps
