'''
this code is using to generate the point cloud data for the point net
import two module datageneration.py and pointcloudgen.py

author: Tcy
Email: tsengcy0411@gmail.com

'''

import argparse
from Astar import AstarMap
from pointcloudgen import genPointUniform_np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description='this code is using to generate point cloud data for pointnet'
    parser.add_argument("--path", help="path to data set", type=str, default="./dataset/")
    parser.add_argument("--type", help="the generation type of the data", type=str, default="train")
    parser.add_argument("--mapstart", help="the start number of the seed to generate the map", type=int, default=0)
    parser.add_argument("--nummap", help="number of map will be generate in this code", type=int, default=2)
    parser.add_argument("--numnode", help="number of pair of start and end will generate", type=int, default=10)
    parser.add_argument("--points", help="number of point that need to be generated", type=int, default=2048)

    args = parser.parse_args()
    width = 224
    height = 224
    path = f'{args.path}{args.type}/'
    print(f'---parameter---\n'
          +f'path to save: {args.path}\n'
          +f'gen type: {args.type}\n'
          +f'map start number: {args.mapstart}\n'
          +f'num of map: {args.nummap}\n'
          +f'num of node: {args.numnode}\n'
          +f'num of points: {args.points}\n'
          +f'path: {path}')
    
    for i in range(args.mapstart, args.nummap+args.mapstart):
        map = AstarMap(width, height, path, i)
        map.initMap()
        map.setclearness(3)
        for j in range(args.numnode):
            map.initNode(j)
            map.clearlistnode()
            check = map.planning()
            if(check):
                genPointUniform_np(map.getinputMap(), map.getresultMap(), map.getshowMap(), 
                                   f'{path}pointcloud/pointcloud_{i}_{j}.csv', 
                                   f'{path}pointcloudimage/pointcloudimage_{i}_{j}.png',args.points,
                                   map.getStart(), map.getGoal())
    print("end")

    