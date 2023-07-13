import os
import cv2
import numpy as np
import torch

def crop_cone_view(board, curr=(4,2), ratio=1, unknown=ord('X')):
    C, H, W = board.shape
    xs = np.arange(W//2)  + 1
    grad1 = -2 * (H * ratio) / W
    b1 = H * ratio
    ys1 = grad1 * xs + b1
    h, w = curr
    board[:, h+1:] = unknown
    for i in range(W//2):
        val = min(-1, -int(ys1[i])-(H-h-1))
        board[:, val:, i] = unknown
        board[:, val:, -i-1] = unknown 
    board = repair(board, board, unknown)
    return board


def _repair(env, h, w, idx, mode='cross'):
    """
        Fix some errors in observation due to discrete space
    """
    if env[h, w] != 0:
        return env
    env[h, w] = idx
    H, W = env.shape
    if mode == 'square':
        for dh in [-1, 0, 1]:
            for dw in [-1, 0, 1]:
                if dh == 0 and dw == 0:
                    continue
                if h + dh < 0 or h + dh == H or w + dw < 0 or w + dw == W:
                    continue
                if env[h+dh, w+dw] == 0:
                    env = _repair(env, h+dh, w+dw, idx)
                if env[h+dh, w+dw] == -1:
                    env[h+dh, w+dw] = 100
    elif mode == 'cross':
#        for dh, dw in zip([-1, +1, 0, 0], [0, 0, -1, +1]):
        for dh, dw in zip([-1, 0, 0], [0, -1, +1]):
            if h + dh < 0 or h + dh == H or w + dw < 0 or w + dw == W:
                continue
            if env[h+dh, w+dw] == 0:
                env = _repair(env, h+dh, w+dw, idx)
            elif env[h+dh, w+dw] == -1: # wall
                env[h+dh, w+dw] = 100
    return env


def repair(board, original_board, block):
    mask = -(board.sum(0) > 0).astype(int)
    H, W = mask.shape
    mask = _repair(mask, H-1, W//2, 1, 'cross')
    # occluded area
    mask = mask <= 0 # 0: empty but visible via slit between wall in a diagonal manner, -1: occulded by wall
    board[:,mask] = block

    return board


def update_blocking(board, original_board, obstacles=[61, 37], unknown=ord('X')):
    C, H, W = board.shape
    w = W // 2
#    loc_obstacles = np.sum([((board==code).sum(0) == 3) for code in obstacles], axis=0) > 0
    loc_blocked = (board==unknown).sum(0) == board.shape[0]
    loc_empty = (board==0).sum(0) == board.shape[0]
    loc_obstacles = np.logical_and(np.logical_not(loc_empty), np.logical_not(loc_blocked))
    row, col = loc_obstacles.nonzero()
    row = row[::-1]
    col = col[::-1]
    for r, c in zip(row, col):
        if r == 0:
            continue
        if c == W//2:
            board[:, :r, c] = unknown
            continue
        if c < W//2:
            xs = np.arange(c+1)
            grad1 = (H  - r) / (W/2 - (c+1))
            grad2 = (H  - (r+1)) / (W/2 - c)
            ys1 = H - 1 - grad1 * (W//2 - xs)
            ys2 = H - 1 - grad2 * (W//2 - xs)
        elif c > W//2:
            xs = np.arange(start=c, stop=W)
            grad1 = (H - r) / (W/2 - c)
            grad2 = (H - (r+1)) / (W/2 - (c+1))
            ys1 = grad1 * (xs-W//2) + H - 1
            ys2 = grad2 * (xs-W//2) + H - 1
        ys1 = np.ceil(ys1.clip(min=0, max=r))
        ys2 = np.ceil(ys2.clip(min=0, max=r))

        for x, y1, y2 in zip(xs, ys1, ys2):
            board[:, int(y1):int(y2), x] = unknown
    board = repair(board, original_board, unknown)
    return board

def get_obs(h, w, d, maze, size=(5,5), cone_view=True, blocking=True, ratio=1, unknown_marker=ord('X')):
    """
        Get Observation from the map

    """
    H, W = size
    dh1s = [-H+1, +0, -(W//2), -(W//2)]
    dh2s = [+1, H, W-(W//2), W-(W//2)]
    dw1s = [-(W//2), -(W//2), -H+1, +0]
    dw2s = [W-(W//2), W-(W//2), +1, H]

    rotation_factor = [0, 2, 3, 1]
    dh1, dh2 = dh1s[d], dh2s[d]
    dw1, dw2 = dw1s[d], dw2s[d]
    rot = rotation_factor[d]
    obs = padding_obs(h+dh1, h+dh2, w+dw1, w+dw2, maze, unknown_marker)
    obs = np.rot90(obs, rotation_factor[d], axes=(-2, -1))
    original_obs = obs.copy()
    if blocking:
        obs = update_blocking(obs, original_obs, unknown=unknown_marker)
    if cone_view:
        obs = crop_cone_view(obs, (size[0]-1, size[1]//2), ratio=ratio, unknown=unknown_marker)
    return obs


def padding_obs(h1, h2, w1, w2, maze, padding_value=ord('X')):
    shape = maze.shape
    H, W = shape[-2:]
    if len(shape) == 3:
        C = shape[0]
        obs = np.ones((C, h2-h1, w2-w1)) * padding_value
    else:
        obs = np.ones((h2-h1, w2-w1)) * padding_value

    st_h = -h1 if h1<0 else 0
    st_w = -w1 if w1<0 else 0
    ed_h = H-h2 if h2>H else h2-h1
    ed_w = W-w2 if w2>W else w2-w1

    h1 = max(h1, 0)
    w1 = max(w1, 0)
    h2 = min(h2, H)
    w2 = min(w2, W)

    if len(shape) == 3:
        try:
            obs[:, st_h:ed_h, st_w:ed_w] = maze[:, h1:h2, w1:w2]
        except:
            breakpoint()
    else:
        obs[st_h:ed_h, st_w:ed_w] = maze[h1:h2, w1:w2]
    return obs
