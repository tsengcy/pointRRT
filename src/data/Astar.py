"""
author: 曾建堯
objective: this code is going to genertate the dataset for the point RRT, 
output: 
image of the enviroment (224x224)
groundturth is A star with 8 neignbor
sample N point
"""
import cv2
import numpy as np
import random
from scipy import signal

class node():
    def __init__(self, _pos: np.array, _goal: np.array, _id: int, _width: int, _value: int):
        self.pos = _pos
        self.gvalue = 100000
        self.hvalue = np.linalg.norm(_goal - self.pos, ord=2)
        self.id = _id
        self.width = _width
        self.value = _value
    
    def __lt__(self, other):
        return self.fvalue < other.fvalue
    
    def __str__ (self): 
        return f"position {self.pos}"
    
    def __hash__(self):
        return self.id
    
    def __eq__(self, other):
        return self.id == other.id
    
    def getneighbor(self)->list:
        return [self.id+1, self.id-1, self.id-self.width-1, self.id-self.width, self.id-self.width+1, self.id+self.width-1, self.id+self.width, self.id+self.width+1]
    
    def extend(self, _par)->bool:
        if(self.gvalue > _par.getdis(self.pos) and self.value != 1):
            self.gvalue = _par.getdis(self.pos)
            self.fvalue = self.gvalue + self.hvalue
            self.parent = _par.id
            return True
        else:
            return False

    def getpos(self):
        return self.pos
           
    def getdis(self, _node)->float:
        return self.gvalue + np.linalg.norm(self.pos - _node, ord=2)
    
    def getid(self):
        return self.id
    
    def setAsStart(self):
        self.parent = -1
        self.gvalue = 0
    
    def getparent(self):
        return self.parent
    
    def getvalue(self):
        return self.value

class AstarMap():
    def __init__(self, _width: int, _height: int, _path: str, _map: int):
        self.width = _width
        self.height = _height
        self.path = _path
        self.mapnumber = _map
        random.seed(self.mapnumber)
        self.initMap()
    
    def initMap(self):
        self.mapobs = np.zeros((self.height, self.width))
        self.mapobs[:,  0] = 1
        self.mapobs[:, -1] = 1
        self.mapobs[0,  :] = 1
        self.mapobs[-1, :] = 1

        for _ in range(50):
            bwidth = random.randint(1, 30)
            bheight = random.randint(1, 30)
            x = random.randint(1, self.width-1-bwidth)
            y = random.randint(1, self.height-1-bheight)

            self.mapobs[y:y+bheight, x:x+bwidth] = 1

        self.blankmap = np.zeros((self.height, self.width, 3))
        self.blankmap[:, :, 0] = self.mapobs
        self.blankmap[:, :, 1] = self.mapobs
        self.blankmap[:, :, 2] = self.mapobs
        self.blankmap = np.where(self.blankmap==1, 0, 255)

    def setclearness(self, _clearness):
        """
        this function use the diation method in cv to extend the obstacle size
        """
        self.clearness = _clearness
        kernel = np.array([[1,1,1], [1,1,1], [1,1,1]])
        self.map = self.mapobs.copy()
        for _ in range(self.clearness):
            self.map = signal.convolve2d(self.map, kernel, "same")
        self.map = np.where(self.map>0, 1, 0)

    def clearlistnode(self):
        self.listnode = []
        for i in range(self.width * self.height):
            y = i//self.width
            x = i%self.width
            self.listnode.append(node(np.array([x, y]), self.goal, i, self.width, self.mapobs[y, x]))
        self.listnode[self.width*self.start[1] + self.start[0]].setAsStart()
        self.goalnode = self.listnode[self.width * self.goal[1] + self.goal[0]]

    def initNode(self, count):
        """
        this node is using for initalizing the start and goal node for planning
        """

        self.mapstart = np.zeros((self.height, self.width))
        self.mapgoal = np.zeros((self.height, self.width))

        while True:
            self.start = np.array([random.randint(1, self.width-2), random.randint(1, self.height-2)])
            if(self.mapobs[self.start[1], self.start[0]]==0):
                break
        while True:
            self.goal = np.array([random.randint(1, self.width-2), random.randint(1, self.height-2)])
            if(self.mapobs[self.goal[1], self.goal[0]]==0 and not (self.goal==self.start).any()):
                break
        self.mapstart[self.start[1], self.start[0]] = 1
        self.mapgoal[self.goal[1], self.goal[0]] = 1
        # cv2.imwrite(f"{self.path}input/input2_.png", self.mapstart*255)

        print(f"start {self.start[1]}, {self.start[0]}")
        self.Nodenumber = count
    
    def diations(self, num):
        kernel = np.array([[0,1,0], [1,1,1], [0,1,0]])
        # self.map = self.mapobs.copy()
        for _ in range(num):
            self.resultmap = signal.convolve2d(self.resultmap, kernel, "same")
        self.resultmap = np.where(self.resultmap>0, 1, 0)
        self.resultmap = np.where(self.mapobs==0, self.resultmap, 0)

    def planning(self):
        self.resultmap = np.zeros((self.height, self.width))
        
        openset = {self.listnode[self.width*self.start[1]+self.start[0]]}
        flag = False

        self.input = np.zeros((self.height, self.width, 3))
        # input[:,:, 0] = np.where(self.mapobs==0, 255, 0)
        self.input[:,:, 2] = np.where(np.logical_xor(self.mapobs==1, np.logical_xor(self.mapstart==1, self.mapgoal==1)), 0, 255)
        self.input[:,:, 1] = np.where(np.logical_and(self.mapgoal==0, self.mapobs==0), 255, 0)
        self.input[:,:, 0] = np.where(np.logical_and(self.mapobs==0, self.mapstart==0), 255, 0)

        while openset:
            open = sorted(openset)
            cur = open[0]
            openset.remove(cur)
            neighbor = cur.getneighbor()
            for nnode in neighbor:
                if(self.listnode[nnode].extend(cur)):
                    openset.add(self.listnode[nnode])
                    if(self.listnode[nnode] == self.goalnode):
                        openset.clear()
                        flag = True
                        break
        
        

        self.maptoshow = self.input.copy()

        cv2.imwrite(f"{self.path}input/input_{self.mapnumber}_{self.Nodenumber}.png", self.input)
        if(flag):
            currentid = self.goalnode.getid()
            while True:
                y = currentid//self.width
                x = currentid%self.width
                self.resultmap[y, x] = 1
                self.maptoshow[y, x, 0:2] = 0
                currentid = self.listnode[currentid].getparent()

                if(currentid == -1): 
                    break
            self.diations(5)
            cv2.imwrite(f"{self.path}result/result_{self.mapnumber}_{self.Nodenumber}.png", self.resultmap)
            cv2.imwrite(f"{self.path}show/show_{self.mapnumber}_{self.Nodenumber}.png", self.maptoshow)
            self.maptoshow[:,:,0] = np.where(self.resultmap==0, self.maptoshow[:,:,0], 0)
            self.maptoshow[:,:,1] = np.where(self.resultmap==0, self.maptoshow[:,:,1], 0)
            cv2.imwrite(f"{self.path}showgroundturth/showgroundturth_{self.mapnumber}_{self.Nodenumber}.png", self.maptoshow)
            return True
        else:
            print("no path")
            # cv2.imwrite(f"{self.path}result/result_{self.mapnumber}_{self.Nodenumber}.png", self.resultmap)
            # cv2.imwrite(f"{self.path}show/show_{self.mapnumber}_{self.Nodenumber}.png", self.maptoshow)

            return False
        
    def getinputMap(self)->np.ndarray: 
        # print(input.shape)
        return self.input
    def getshowMap(self)->np.ndarray:
        return self.maptoshow
    def getresultMap(self)->np.ndarray:
        return self.resultmap
    
    def getinputPath(self)->str:
        return f"{self.path}input/input_{self.mapnumber}_{self.Nodenumber}.png"
    def getshowPath(self)->str:
        return f"{self.path}showgroundturth/showgroundturth_{self.mapnumber}_{self.Nodenumber}.png"
    def getresultPath(self)->str:
        return f"{self.path}result/result_{self.mapnumber}_{self.Nodenumber}.png"
        
if __name__=="__main__":
    width = 224
    height = 224
    path = "./dataset/"
    astar = AstarMap(width, height, path, 0)
    astar.initMap()
    astar.initNode(0)
    astar.clearlistnode()
    astar.planning()

    astar.initNode(1)
    astar.clearlistnode()
    astar.planning()


        

