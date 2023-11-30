from matplotlib import pyplot as plt
import numpy as np
import cv2
import random
import math
from PIL import Image

def genPointUniform(_pathobs: str, _pathresult: str, _pathshow: str, _pathPC: str, _num: int):
    """
    return the list 
    """
    imgInput = cv2.imread(_pathobs)
    imgshow = Image.open(_pathshow)
    # cv2.imshow("img", imgInput)
    # cv2.waitKey(0)
    imgResult = cv2.imread(_pathresult)
    print(imgInput.shape)
    height, width, _ = imgInput.shape
    points = np.zeros((_num, 4))
    count = 0
    axlist = []
    aylist = []
    bxlist = []
    bylist = []
    while True:
        if count == _num:
            break
        x = random.uniform(0, width-1)
        y = random.uniform(0, height-1)
        if(imgInput[round(y), round(x), 2] != 0):
            if(imgResult[round(y), round(x), 0] == 0):
                points[count, :] = np.array([x, y, 0, 1])
                axlist.append(x)
                aylist.append(y)
            else:
                points[count, :] = np.array([x, y, 0, 0])
                bxlist.append(x)
                bylist.append(y)
            count += 1
            # print(count)
    # print(points)
    # fig, ax = plt.subplots()
    plt.imshow(imgshow)
    plt.scatter(axlist, aylist, c='tab:orange', s=1)
    plt.scatter(bxlist, bylist, c='tab:blue', s=1)
    plt.savefig("img.png")
    plt.show()

    np.savetxt(f'{_pathPC}', points, delimiter=",")
    
def genPointUniform_np(_obs: np.ndarray, _result: np.ndarray, _show: np.ndarray, _pathPC: str, _pathPCimg: str, _num: int, start: np.ndarray, goal: np.ndarray):
    """
    return the list 
    """
    # imgInput = cv2.imread(_pathobs)
    # imgshow = Image.open(_pathshow)
    # cv2.imshow("img", imgInput)
    # cv2.waitKey(0)
    # imgResult = cv2.imread(_pathresult)
    # print(imgInput.shape)
    _obs = _obs.astype('uint8')
    _show = _show[:,:,[2,1,0]]
    height, width, _ = _obs.shape
    points = np.zeros((_num, 5))
    count = 0
    axlist = []
    aylist = []
    bxlist = []
    bylist = []
    while True:
        if count == _num:
            break
        x = random.uniform(0, width-1)
        y = random.uniform(0, height-1)
        if(_obs[round(y), round(x), 2] != 0):
            if(_result[round(y), round(x)] == 0):
                points[count, :] = np.array([x-width/2, y-height/2, 0, 0, 0])
                axlist.append(x)
                aylist.append(y)
            else:
                if(np.linalg.norm(start-np.array([x,y]), ord=2)>5):
                    points[count, :] = np.array([x-width/2, y-height/2, 0, 1, 1])
                elif(np.linalg.norm(goal-np.array([x,y]), ord=2)>5):
                    points[count, :] = np.array([x-width/2, y-height/2, 0, 2, 1])
                else:
                    points[count, :] = np.array([x-width/2, y-height/2, 0, 0, 1])
                bxlist.append(x)
                bylist.append(y)
            count += 1
            # print(count)
    # print(points)
    # fig, ax = plt.subplots()
    plt.imshow(_show.astype('uint8'))
    plt.scatter(axlist, aylist, c='tab:orange', s=1)
    plt.scatter(bxlist, bylist, c='tab:blue', s=1)
    plt.savefig(_pathPCimg)
    plt.close()
    # plt.show()

    np.savetxt(f'{_pathPC}', points, delimiter=",")

if __name__=="__main__":
    pathInput = "./dataset/input/input_0_0.png"
    pathresult = "./dataset/result/result_0_0.png"
    pathShow = "./dataset/showgroundturth/showgroundturth_0_0.png"
    pathPC = "./dataset/pointcloud/pointcloud_0_0.csv"
    genPointUniform(pathInput, pathresult, pathShow, pathPC, 2048)



