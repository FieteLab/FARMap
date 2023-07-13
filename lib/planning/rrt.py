import sys
import cv2
import numpy as np
import math
import random
import argparse
import os
import torch

def visualize(paths, connections, img):
    connections = np.asarray(connections, dtype=int)
    paths = np.asarray(paths, dtype=int)
    cv2.circle(img, tuple(paths[0][::-1]), 5,(0,0,255),thickness=3, lineType=8)
    cv2.circle(img, tuple(paths[-1][::-1]), 5,(0,0,255),thickness=3, lineType=8)
    count = 0
    for connection in connections:
        x1, y1, x2, y2 = connection
        cv2.circle(img, (y1, x1), 2,(0,0,255),thickness=3, lineType=8)
        cv2.line(img, (y1, x1), (y2, x2), (0,255,0), thickness=1, lineType=8)
        cv2.imwrite("media/{:03}.jpg".format(count),img)
        count += 1

    for i in range(len(paths)-1):
        cv2.line(img, tuple(paths[i][::-1]), tuple(paths[i+1][::-1]), (255,0,0), thickness=2, lineType=8)

    cv2.imwrite("out.jpg",img)
    cv2.imwrite("media/{:03}.jpg".format(count+1), img)


class Nodes:
    """
        Class to store the RRT graph
    """
    def __init__(self, pos, path=[]):
        self.pos = pos
        self.path = path

    def add_path(self, path):
        self.path.append(path)
    
    def set_path(self, path):
        self.path = path.copy()



def get_dist(pos1, pos2, l1=False):
    x1, y1 = pos1
    x2, y2 = pos2

    if l1:
        dist = abs(x2-x1) + abs(y2-y1)
    else:
        dist = math.sqrt(((x1-x2)**2)+((y1-y2)**2))
    return dist

# generate a random point in the image space
def rnd_point(h,l):
    new_y = random.randint(0, l)
    new_x = random.randint(0, h)
    return [new_x,new_y]


def straight_point(xs, ys, H, W):
    idx = random.randint(0, len(xs)-1)
    st_x = xs[idx]
    st_y = ys[idx]
    
    d = random.randint(0, 3)
    dx = [-1, +1, 0, 0][d]
    dy = [0, 0,  -1, +1][d]
    new_y = random.randint(0, W-1)
    new_x = random.randint(0, H-1)

    return st_x + dx * new_x, st_y + dy * new_y

def straight_point(xs, ys, H, W):
    xs = np.unique(xs)
    ys = np.unique(ys)
    idx = random.randint(0, len(xs)+len(ys)-1)
    if idx >= len(xs):
        st_x = random.randint(0, H-1)
        st_y = ys[idx-len(xs)]
    else:
        st_x = xs[idx]
        st_y = random.randint(0, W-1)
    
    return st_x, st_y
    

def is_collision(pos1, pos2, img, step_size, interval=1/100):
    """
        Check whether collison is happened or not.
    """
    if step_size == 1:
        return img[int(pos1[0]), int(pos1[1])] == 0
    H, W = img.shape
    x1, y1 = pos1[:2]
    x2, y2 = pos2[:2]

    # out of range
    occupied = []
    if x1 != x2:
        x = np.arange(x1, x2, interval * (x2-x1))
        y = ((y2-y1) / (x2-x1)) * (x-x1) + y1
    else:
        y = np.arange(y1, y2, interval * (y2-y1))
        x = np.asarray([x1 for _ in y])
    
    pos = np.unique(np.stack([x, y]).astype(int), axis=-1).T
    for x, y in pos:
        if img[x, y] == 0:
            return True
    return False

def check_head(pos, end, img):
    head = end[-1]
    dh = end[0] - pos[0]
    dw = end[1] - pos[1]

    up_down = dh > 0
    left_right = dw > 0

    return True

def RRT(img, start, end, step_size=1, discrete_head=True):
    """
        Run RRT Algorithm
        Input:
            img: 2D numpy array of the image
            start: start point
            end: end point
            step_size: step size of the RRT
            discrete_head: if True, the head of the robot is discrete
        Output:
            path: list of points in the path
            connections: list of connections in the graph
    """
    H, W = img.shape
    if type(start) is tuple:
        start = list(start)
    if type(end) is tuple:
        end = list(end)

    node_list = []
    node_list.append(Nodes(start, [start]))

    # display start and end
    connections = []
    done = False

    xs = [start[0]]
    ys = [start[1]]
    max_iter = 1000
    step = 0
    while not done and step < max_iter:
        step += 1
        if discrete_head:
            nx, ny = straight_point(xs, ys, H, W)
        else:
            nx,ny = rnd_point(H,W)

        dist = sys.maxsize
        for i, node in enumerate(node_list):
            if discrete_head:
                if nx != node.pos[0] and ny != node.pos[1]:
                    continue
            d = get_dist([nx, ny], node.pos, discrete_head)
            if d < dist:
                dist = d
                nearest_node = node
        nearest_pos = nearest_node.pos

        x2, y2 = nearest_pos
        theta = math.atan2(nx-x2, ny-y2)

        x = x2 + step_size * np.sin(theta)
        y = y2 + step_size * np.cos(theta)
        next_pos = [x, y]
        
        # Out of Range
        if x < 0 or x >=  H or y < 0 or y >= W:
            continue
        if is_collision(next_pos, nearest_pos, img, step_size):
            continue

        xs.append(x)
        ys.append(y)

        new_node = Nodes(next_pos, nearest_node.path.copy())
        new_node.add_path(next_pos)
        node_list.append(new_node)
        connections.append([next_pos[0], next_pos[1], nearest_pos[0], nearest_pos[1]])
        done = not is_collision(next_pos, end, img, step_size)
        if discrete_head:
            done = done and check_head(next_pos, end, img)
    if not done:
        return None, None
    paths = new_node.path
    connections.append([end[0], end[1], next_pos[0], next_pos[1]])
    paths.append(end)
    return paths, connections



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Below are the params:')
    parser.add_argument('-p', type=str, default='world2.png',metavar='ImagePath', action='store', dest='imagePath',
                    help='Path of the image containing mazes')
    parser.add_argument('-s', type=int, default=10,metavar='Stepsize', action='store', dest='stepSize',
                    help='Step-size to be used for RRT branches')
    parser.add_argument('-start', type=int, default=[20,20], metavar='startCoord', dest='start', nargs='+',
                    help='Starting position in the maze')
    parser.add_argument('-stop', type=int, default=[250, 450], metavar='stopCoord', dest='stop', nargs='+',
                    help='End position in the maze')
    parser.add_argument('-selectPoint', help='Select start and end points from figure', action='store_true')

    args = parser.parse_args()

    # remove previously stored data
    try:
      os.system("rm -rf media")
    except:
      print("Dir already clean")
    os.mkdir("media")

    img = cv2.imread(args.imagePath,0) # load grayscale maze image
    img2 = cv2.imread(args.imagePath) # load colored maze image
    start = tuple(args.start) #(20,20) # starting coordinate
    end = tuple(args.stop) #(450,250) # target coordinate
    stepSize = args.stepSize # stepsize for RRT

    end = [250, 450, 2]

    coordinates=[]
    # run the RRT algorithm 
    paths, connections = RRT(img, start, end, stepSize, True)
    visualize(paths, connections, img2)
