import os
import numpy as np

import torch

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split

def setSeed(_seed = 0):
    np.random.seed(_seed)
    torch.manual_seed(_seed)

def getDataPath(_path):
    listPathPoint, listPathShow = [], []

    dataPath = _path
    for folder in tqdm(sorted(os.listdir(dataPath))):
        if(folder == "pointcloud"):
            for sequence in tqdm(sorted(os.listdir(f'{dataPath}/{folder}'))):
                fullpath = f'{dataPath}/{folder}/{sequence}'
                listPathPoint.append(fullpath)
        elif(folder == "showgroundturth"):
            for sequence in tqdm(sorted(os.listdir(f'{dataPath}/{folder}'))):
                fullpath = f'{dataPath}/{folder}/{sequence}'
                listPathShow.append(fullpath)
    print(f'Number of pointcloud data: {len(listPathPoint)}, Number of pointcloud image: {len(listPathShow)}')
    return np.array(listPathPoint), np.array(listPathShow)

def getDataLoader(dataDir, batchSize, split, valid_rate=0.2):
    pathPoint, pathShow = getDataPath(dataDir)
    dataSet = pointPathDataSet(pathPoint)
    if(split == "train"):
        numValid = round(len(dataSet) * valid_rate)
        numTrain = len(dataSet - numValid)
        trainset, validset = random_split(dataSet, [numTrain, numValid])
        trainLoader = DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        validLoader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
        return trainLoader, validLoader
    elif(split == "test"):
        testLoader = DataLoader(dataSet, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
        return testLoader
class pointPathDataSet(Dataset):
    def __init__(self, _listPathPoint) -> None:
        super().__init__()
        self.path = _listPathPoint
        # self.pathshow = _listPathShow
    def __len__(self)->int:
        return len(self.path)
    def __getitem__(self, idx):
        data = np.load(self.path[idx])
        point = data[:, 0:3]
        label = data[:, 3]
        return {
            'point': point,
            'label': label
        }

        