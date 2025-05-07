""" Module dedicated to loading files over different point cloud formats into torch"""

import torch
import laspy
import numpy as np

def load_las(filename:str):
    points = [[]]
    las = laspy.read(filename)
    points_float = las.xyz.astype(np.float32)
    for point in points_float:
         points[0].append(point)
    return torch.tensor(points)
    
def load_xyz(filename):
    points = [[]]
    with open(f"{filename}",'r') as f:
        while True:
            line = f.readline()
            if line == "":
                break
            xyz = line.replace('\n', '').split(' ')
            xyz = xyz[0:3]
            point = []
            for el in xyz:
                point.append(float(el))
            
            points[0].append(point)
    return torch.tensor(points)