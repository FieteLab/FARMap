import argparse
import os
import numpy as np
import json
import torch

from lib.utils import get_map, get_obs, set_seed, size_visible_area, get_start_loc, discovery, colorize_maze, enlarge_map, get_color_map
from lib.agent import FrontierAgent

import matplotlib.pyplot as plt
import pickle
import wandb

def prepare_observation(h, w, d, env, size, cone_view, blocking, ratio, observed_map, P, use_cuda, original_env=None, colors=None, unknown_marker=ord('X')):
    if original_env is not None:
        _env = np.copy(original_env)
        _env = colorize_maze(_env, colors=colors)
        env = enlarge_map(_env, (5,5), scale=3, use_start_pos=False)[0]

    obs = get_obs(int(h), int(w), int(d), env, size=size, cone_view=cone_view, 
            blocking=blocking, ratio=ratio, unknown_marker=unknown_marker)
    # update discovered regions
    observed_map = discovery(int(h), int(w), int(d), obs, observed_map, env, P)
    obs = torch.from_numpy(obs.copy()).float() / 255
    if use_cuda:
        obs = obs.cuda()
    return obs, observed_map



def explore(curr_loc, num_steps, agent_type, epsilon=5, planning=False, stochastic=True, save_dir='./', rho=2, gamma=0.9, frag_mode='z', ratio=0.25, obs_size=(15,15), verbose=True, unknown_marker=ord('X'), use_cuda=False):
    """
        Run actual experiments.
        curr_loc: starting location of the environment.
        num_steps: the episode length.
        agent_type: agent type [Frontier, FarMap]
        epsilon: a hyperparameter for finding desirable map in LTM.
        planning: use planner instead of a single step prediction.
        stochastic: use weighted sampling.
        rho: fragmentation threshold.
        gamma: exponential decaying factor.
        frag_mode: the criteria for fragmentation (z-score or ratio of curr_surprsial / mean).
        ratio: FOV, the length of invissible region. e.g., ratio = 0.25, L = 16 => 4 pixels are invisible due to restricted FOV which is ~130 degree.
    """
    # observation size. (Channel, 5*scale, 5*scale)
    image_shape = (3, obs_size[0], obs_size[1])
    fragmentation = agent_type == 'FarMap'
    blocking = True
    cone_view = True
    use_rrt = False
    name = agent_type

    if stochastic:
        name = name + '_stochastic'
    else:
        name = name + '_deterministic'
    if original_env is not None:
        target_map = original_env[0] == ord('%')
        N = target_map.sum()
        colors = get_color_map(N+1)[1:]
    else:
        colors = None

    print(name)

    print('CUDA:', use_cuda)

    agent = FrontierAgent(image_shape, decaying_factor=gamma, stochastic=stochastic, is_planning=planning, use_rrt=use_rrt, fragmentation=fragmentation, epsilon=epsilon, rho=rho, frag_mode=mode, unknown_marker=args.unknown_marker)

    C, H, W = env.shape
    P = max(image_shape[1:])
    observed_map = np.zeros((H + 2*P, W + 2*P))

    trajs = []
    subgoals = []
    h, w, d = curr_loc
    h = int(h)
    w = int(w)
    trajs.append([h, w, d])
    subgoals.append([h, w, d])

    print(trajs[-1])
    area = size_visible_area(env)
    observed_maps = []
    step = 0
    steps = []
    conf_maps = []
    
    goal_count = 0
    start_idx = 0
    break_flag = False

    while True:
        # get observation
        # obs: observation
        # observed_map: global map that marks currently observing regions.
        obs, observed_map = prepare_observation(int(h), int(w), int(d), env, image_shape[1:],
                            cone_view, blocking, ratio, observed_map, P, use_cuda, unknown_marker=unknown_marker)

        is_mismatch = obs.sum(0)[-1, 7] > 0
        if is_mismatch:
            if verbose:
                print("MISMATCH\n\n\n")
            obs = prev_obs
            d = prev_d
            # flush prev ones;
            if hasattr(agent, 'scores'):
                agent.scores = agent.scores[:-1]
            if hasattr(agent, 'memory_size'):
                agent.memory_size = agent.memory_size[:-1]
        else:
            steps.append(step)
            observed_maps.append(observed_map[P:-P, P:-P].copy())

        subgoal = agent.step(obs.unsqueeze(0), d) # find subgoal (X, Y, Head Direction)

        if subgoal is None: # the agent already explored the entire space
            break_flag = True
            break
        if type(subgoal) is int and subgoal == 2147483647: # something wrong in process
            break_flag = True
            break

        # intermediate locations
        plans = agent.plan()

        step += 1
        goal_count += 1

        observations = []
        if 'FarMap' == agent_type: # for visualizing confidence map.
            conf_maps.append(agent.get_confidence_map())

        # move agent following the plan
        for plan in plans:
            ph, pw, pd = plan
            obs, observed_map = prepare_observation(int(h+ph), int(w+pw), pd, env, image_shape[1:],
                                cone_view, blocking, ratio, observed_map, P, use_cuda, unknown_marker=unknown_marker)

            is_mismatch = obs.sum(0)[-1, 7] > 0
            if is_mismatch: # the agent is on the wall.... It should not happen.
                if verbose:
                    print("MISMATCH\n\n\n")
                break
            if step >= num_steps: # it exceeds the maximum number of steps.
                break

            observations.append(obs)
            observed_maps.append(observed_map[P:-P, P:-P].copy())
            steps.append(step)
            trajs.append([h+ph,w+pw,pd])
            subgoals.append([h+subgoal[0], w+subgoal[1], subgoal[2]])
            step += 1
            prev_obs = obs

        plans = plans[:len(observations)] # only recorded actually acted action.
        locs = []
        for plan in plans:
            locs.append([h + plan[0], w + plan[1]])
        confidence_maps = agent.memory_update(observations, plans)
        if confidence_maps is not None:
            conf_maps += confidence_maps

        if step >= num_steps: # exceed the number of steps.
            break

        if is_mismatch:
            if len(plans) > 0:
                h = h + plans[-1][0]
                w = w + plans[-1][1]
                d = plans[-1][2]
            if hasattr(agent, 'saved_stm'):
                agent.saved_stm = None
        else:
            prev_d = d
            prev_obs = obs
            # the model get stuck.
            if subgoal[0] == 0 and subgoal[1] == 0 and d == subgoal[2]:
                print(subgoal)
                if step <= 1:
                    return
                break
            h = h + subgoal[0]
            w = w + subgoal[1]
            d = subgoal[2]
            trajs.append([h,w,d])
            subgoals.append([h,w,d])

        observed_area = (observed_map>0).sum()
        if verbose:
            print('{:03}: X: {}, Y: {}, D: {}, Action: {}, Discovered: [step] {} / {} ({:.3}) [area] {} / {} ({:.3})'.format(goal_count+1, h, w, d, subgoal, observed_area,  step, observed_area/step, observed_area, area, observed_area / area))

    scores = getattr(agent, 'scores', [])
    remaps = getattr(agent, 'remap_loc', [])
    recalls = getattr(agent, 'recall_loc', [])
    ltm_subgoals = getattr(agent, 'ltm_subgoal_loc', [])
    memory_sizes = getattr(agent, 'memory_size', [])

    if break_flag: # if the agent already explores the entire space or there is an error. 
        if verbose:
            print("NO WAY TO GO")
        while step < num_steps:
            memory_sizes.append(memory_sizes[-1])
            steps.append(step)
            observed_maps.append(observed_maps[-1])
            step += 1

    if len(scores) > 0:
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        ax.plot(list(range(len(scores))), scores)
        fig.savefig(os.path.join(save_dir, f'{name}_explore.png'))
        plt.close()

    print(f"{remaps} fragmentation happened\n {recalls} recall happened\n {ltm_subgoals} LTM subgoals")
    observed_area = [np.count_nonzero(observed_map) for observed_map in observed_maps]
    return name, observed_area, steps, trajs, subgoals, remaps, recalls, memory_sizes, [0] + scores, conf_maps




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training parameters')
    parser.add_argument('-use_cuda', action='store_true')
    parser.add_argument('-env', type=int, default=0)
    parser.add_argument('-save_dir', type=str, default='./FarMap_exp')
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-scale', type=int, default=3, help='Scale up the map')
    parser.add_argument('-num_steps', type=int, default=5000, help='The maximum number of steps for exploration.')
    parser.add_argument('-rho', type=float, default=2.0, help='fragmentation threshold')
    parser.add_argument('-mode', type=str, default='z', choices=['z', 'ratio', 'random', 'uniform'])
    parser.add_argument('-gamma', type=float, default=0.9, help='exponential decaying factor in local map')
    parser.add_argument('-epsilon', type=float, default=5, help='hyperparameter for choosing desirable map in LTM')
    parser.add_argument('-agent_type', type=str, default='FarMap', choices=['Frontier', 'FarMap'])
    parser.add_argument('-obs_size', type=int, default=5, help='Scale up the map')
    parser.add_argument('-FOV_ratio', type=float, default=0.25, help='FOV ratio (tan (H*ratio) / (W/2)), 0.25 means 130 degree')
    parser.add_argument('-random_colorization', action='store_true', help='randomly changing wall color')

    # wandb
    parser.add_argument('-wandb', action='store_true')
    parser.add_argument('-proj_name', type=str, default='experiments')

    # Marker in the environment
    parser.add_argument('-empty_marker', type=int, default=0, help='empty marker')
    parser.add_argument('-wall_marker', type=int, default=ord('%'), help='wall (occupied) marker')
    parser.add_argument('-unknown_marker', type=int, default=ord('X'), help='unknown marker')

    args = parser.parse_args()
    gamma = args.gamma
    rho = args.rho
    epsilon = args.epsilon
    mode = args.mode
    ratio = args.FOV_ratio
    obs_size = (args.obs_size, args.obs_size)
    scaled_obs_size = (args.obs_size*args.scale, args.obs_size*args.scale)
    stochastic = True # we will only use stochastic mode.

    if args.seed >= 0:
        seeds = [args.seed]
    else:
        seeds = range(5)

    for seed in seeds:
        set_seed(seed)

        agent_types = [args.agent_type]

        generation = True
        planning = True
        num_steps = args.num_steps
        scale = args.scale
        _save_dir = args.save_dir
        if not os.path.exists(_save_dir):
            os.mkdir(_save_dir)

        env_ids = [args.env]
        original_env_id = args.env

        for env_id in env_ids: # run experiments
            name = f'run_{scale}x_env{env_id}_step{num_steps}_graph_epsilon{epsilon}_rho{rho}_gamma{gamma}_{mode}_ratio{ratio}'
            if not os.path.exists(os.path.join(_save_dir, name)):
                os.mkdir(os.path.join(_save_dir, name))
            save_dir = os.path.join(_save_dir, name, f'seed_{seed}')
            print(f"Seed {seed} Env_id {env_id}")

            env, curr_loc, original_env = get_map(env_id, scale=scale, generation=generation, maze_path='farmap_dataset.pkl', obs_size=obs_size, wall=args.wall_marker, empty=args.empty_marker)
            if not args.random_colorization:
                original_env = None
            size = size_visible_area(env)
            if env_id == -1:
                C, H, W = env.shape
                curr_loc = (H//2, W//2, 0)
            else:
                curr_loc = get_start_loc(curr_loc, env)
            results = []
            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot(111)
            max_num_moves = -1

            fig2 = plt.figure(figsize=(12,8))
            ax2 = fig2.add_subplot(111)

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            print(save_dir)
            exist = False # already exist set up
            for agent_type in agent_types:
                print(save_dir, agent_type)
                if stochastic:
                    name = f'{agent_type}_stochastic'
                else:
                    name = f'{agent_type}_deterministic'
                summary = os.path.join(save_dir, 'wandb/latest-run/files/wandb-summary.json')
                if os.path.exists(summary):
                    with open(summary, 'r') as f:
                        prev_step  =json.load(f).get('step', 0)
                        print(prev_step)
                    if prev_step >= num_steps - 300:
                        print(f'{name} alreday exists in WANDB')
                        exist = True
                        continue


                if args.wandb:
                    config = {'env': args.env, 'env_id': env_id, 'seed': seed, 'scale': scale,
                                'epsilon': epsilon, 'method': agent_type, 'stochastic': stochastic,
                                'size': size, 'max_num_steps': num_steps, 'rho':rho, 'gamma': gamma,
                                'original_env_id': original_env_id}

                    wandb.init(name=f'Env{env_id}_{name}_x{scale}_{epsilon}_rho{rho}_gamma{gamma}_Seed{seed}_{mode}',
                            project=args.proj_name,
                            config=config,
                            dir=save_dir
                            )


                set_seed(seed)
                ret = explore(curr_loc, num_steps, agent_type, epsilon=epsilon, planning=planning, 
                                stochastic=stochastic, save_dir=save_dir, rho=rho, gamma=gamma, frag_mode=mode, ratio=ratio, obs_size=scaled_obs_size, use_cuda=args.use_cuda, unknown_marker=args.unknown_marker)
                if ret is None:
                    continue

                with open(os.path.join(save_dir,f'{ret[0]}_data.pkl'), 'wb') as f:
                    pickle.dump(ret, f)

                name = ret[0]
                num_seens = ret[1]
                num_moves = ret[2]
                results.append(ret)
                ratio = num_seens[-1] / num_moves[-1]
                print(f'{name}: {num_seens[-1]} / {size}  / {num_moves[-1]} ({ratio})')

                max_num_moves = max(max_num_moves, num_moves[-1])
                if args.wandb:
                    logs = {}
                    for i in range(len(num_moves)):
                        logs['step'] = num_moves[i]
                        logs['seen_cell'] = num_seens[i]
                        logs['seen_ratio'] = num_seens[i] / size
                        if len(ret[7]) > 0:
                            logs['memory'] = ret[7][i]
                            logs['memory_ratio'] = ret[7][i] / (env.shape[-2] * env.shape[-1])
                        wandb.log(logs)
                    wandb.finish()

                print(name)
                ax.plot(num_moves, num_seens, label=name)
                ax.legend()
                fig.savefig(os.path.join(save_dir, 'plot.png'))
                if len(ret[7]) > 0:
                    ax2.plot(num_moves, ret[7], label=name)
                    ax2.legend()
                    fig2.savefig(os.path.join(save_dir, 'memory_size.png'))

                print(ret[0], ret[5], ret[6])

            if exist:
                continue

            ax.plot(list(range(max_num_moves)), [size for _ in range(max_num_moves)])
            fig.savefig(os.path.join(save_dir, 'plot.png'))

            for ret in results:
                name = ret[0]
                observed_area = ret[1]
                step = ret[2]
                ratio = observed_area[-1] / step[-1]
                print(f'{name}: {observed_area[-1]} / {size}  / {step[-1]} ({ratio})')

            with open(os.path.join(save_dir,'data.pkl'), 'wb') as f:
                pickle.dump(results, f)


