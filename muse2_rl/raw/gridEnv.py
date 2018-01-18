import numpy as np;
import matplotlib.pyplot as plt;
np.set_printoptions(precision=2)

#N,E,W,S = 0,1,2,3
class Grid(object):
    def __init__(self,length,width,start,terminals,walls=0):
        self.length,self.width = length,width;
        self.grid = np.chararray((length,width));
        self.grid[:]='.'
        self.grid[start]='C'
        self.value = np.random.random((length,width))-0.5
        self.policy = np.random.randint(4,size=(length,width))
        self.qvalue = np.random.random((4,length,width))-0.5;
        for terminal in terminals:
            if(self.isOutOfBound(terminal)):
                raise Exception('Terminal state is out of bound')
            self.grid[terminal]='T'
            self.value[terminal[0]][terminal[1]]=-1.0
        self.start = list(start);
        self.terminals =terminals
        self.curpos =self.start;
        self.t=0;

    def sample(self,action,epsilon=0.5):
        rnum = random.random();
        if (rnum<epsilon):
            return action;
        else:
            rnum2 = np.random.randint(0,3)
            allactions = np.delete(np.arange(4),action);
            return allactions[rnum2];

    def step(self,action,epsilon=0.5):
        self.t+=1;
        for terminal in self.terminals:
            self.value[terminal[0]][terminal[1]]=-1.0
        nextpos=[0,0]
        action = int(action)
        if action>=4 or action<0:
            raise Exception('Action value should be [0,1,2,3]')
        action = self.sample(action,epsilon);
        if action is 0: #North, decrement x
            nextpos[0] = self.curpos[0]-1;
            nextpos[1] = self.curpos[1];
        elif action is 1: #East, increment y
            nextpos[0] = self.curpos[0];
            nextpos[1] = self.curpos[1]+1;
        elif action is 2: #West, decrement y
            nextpos[0] = self.curpos[0];
            nextpos[1] = self.curpos[1]-1;
        elif action is 3: #South, increment x
            nextpos[0] = self.curpos[0]+1;
            nextpos[1] = self.curpos[1];
        #Update Current position
        if self.isOutOfBound(nextpos):
            return (False,-1);
        if self.isTerminal(nextpos):
            self.updatepos(nextpos)
            return (True,1);
            self.t=0;
        self.updatePos(nextpos)
        return (False,0)
    def isOutOfBound(self,nextpos):
        if nextpos[0]<0 or nextpos[0]>=self.length \
          or nextpos[1]<0 or nextpos[1]>=self.width :
            return True;
        return False
    def isTerminal(self,nextpos):
        if tuple(nextpos) in self.terminals:
            return True;
        return False;
    def updatePos(self,nextpos):
        self.grid[self.curpos[0]][self.curpos[1]]='.'
        self.curpos = nextpos;
        self.grid[self.curpos[0]][self.curpos[1]]='C'    
    def display(self):
        for row in self.grid:
            print ('|'+' '.join(row.decode('utf-8'))+'|')

    def displayPolicy(self):
        #NEWS
        arrowDict = {0:'\u2191',1:'\u2192',2:'\u2190',3:'\u2193'}
        for row in self.policy:
            print ('|'+' '.join([arrowDict[r] for r in row])+'|')

    def displayValue(self):
        plt.imshow(self.value,cmap='gray',vmax=1.0,vmin=-1.0)
        plt.show()

    def displayAll(self):
       print ('\nEnvironment at time '+str(self.t))
       self.display()
       print ('\nPolicy')
       self.displayPolicy()
       print ('\nValue')
       self.displayValue();
       print (self.value)

    def reset(self):
       self.t =0;
       self.updatePos(self.start)
       
