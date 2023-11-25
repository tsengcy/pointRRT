from matplotlib import pyplot as plt
import numpy as np
import cv2
import random
import math


def genPoint(_pathobs: str, _pathresult: str, _num: int):
    """
    return the list 
    """
    imgInput = cv2.imread(_pathobs)
    # cv2.imshow("img", imgInput)
    # cv2.waitKey(0)
    imgResult = cv2.imread(_pathresult)
    print(imgInput.shape)
    height, width, _ = imgInput.shape
    points = np.zeros((_num, 3))
    count = 0
    axlist = []
    aylist = []
    bxlist = []
    bylist = []
    while True:
        if count == _num:
            break
        x = random.uniform(width*2/5, width*4/5)
        y = random.uniform(height/2, height)
        if(imgInput[math.floor(y), math.floor(x), 0] == 255):
            if(imgResult[math.floor(y), math.floor(x), 0] == 0):
                points[count, :] = np.array([x, y, 1])
                axlist.append(x)
                aylist.append(height- y)
            else:
                points[count, :] = np.array([x, y, 0])
                bxlist.append(x)
                bylist.append(height - y)
            count += 1
            print(count)
    
    # fig, ax = plt.subplots()
    plt.scatter(axlist, aylist, c='tab:blue', s=1)
    plt.scatter(bxlist, bylist, c='tab:orange', s=1)
    plt.savefig("img.png")
    plt.show()
    

if __name__=="__main__":
    pathInput = "./dataset/input/input_0_0.png"
    pathresult = "./dataset/result/result_0_0.png"
    genPoint(pathInput, pathresult, 2048)



