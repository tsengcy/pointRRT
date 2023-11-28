import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset

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
    

class pointPathDataSet(Dataset):
    def __init__(self, _listPath, _numPoint) -> None:
        super().__init__()
        self.path = _listPath
        self.numPoint = _numPoint
    def __len__(self)->int:
        return len(self.path)
    def __getitem__(self, idx):
        data = np.load(self.path[idx])
        point = data[:, 0:self.numPoint]
        label = data[:, self.numPoint:]
        return {
            'point': point,
            'label': label
        }

        