import numpy as np
from collections import Counter 

def histInt(x,norm=True):
    x = x[~np.isnan(x)]
    c = Counter(x)
    q, w = (list(t) for t in zip(*sorted(zip(list(c.keys()), list(c.values())))))
    q = np.array(q)
    w = np.array(w)
    if norm == True:
        return w/float(sum(w)), q
    else:
        return w, q



def histFloat(x, bins):
    w, q = np.histogram(x, bins)
    dq = q[1] - q[0]
    q = q[:-1] + dq*0.5
    return np.array(w/dq/sum(w)), np.array(q)

def logHist(x,bins):
    #Select only positive values
    x = np.array(x)
    x = x[x>0]
    #Hist of the logs
    wl,ql_ = np.histogram([np.log(xx) for xx in x],bins)
    L = len(wl)
    #Rescale in linear
    q_ = [np.exp(qq) for qq in ql_]
    dq = [q_[i+1]-q_[i]       for i in range(L)]
    q  = [0.5*(q_[i+1]+q_[i]) for i in range(L)]
    #Normalize
    invNorm = float(sum(wl))**(-1);
    w = [wl[i]*invNorm/dq[i] for i in range(L)]
    return np.array(w), np.array(q) ,dq
    

"""
def cumulative(t):
    L = float(len(t))
    x = np.array(sorted(t))
    range_L = np.arange(1,L+1)
    y = range_L/L
    return x,y
"""

def cumulative(t):
    x = sorted(t)
    y2 = range(1, len(x) + 1)
    y = [yy / float(len(x)) for yy in y2]
    return np.array(y), np.array(x)




"""
class AggregateHist(object):
    def __init__(self,xmin,xmax,num,isLog):
        self.isLog = isLog
        self.xmin = xmin
        self.xmax = xmax
        if not self.isLog:
            self.dx = (xmax-xmin)/float(num)
            self.right = np.arange(xmin+self.dx,xmax+self.dx,self.dx)
            self.x = [x-self.dx*0.5 for x in self.right]
        else:
            logdx = (math.log(xmax)-math.log(xmin))/num
            logextr = np.arange(math.log(xmin),math.log(xmax)+logdx,logdx)
            extr = [math.exp(lr) for lr in logextr]
            self.right = extr[1:]
            self.dx =  [extr[i+1]-extr[i] for i in range(len(extr)-1)]
            self.x = [(extr[i+1]+extr[i])*0.5 for i in range(len(extr)-1)]


        self.y = [0]*len(self.x)

    def addValue(self,valueX):
        if valueX >= self.xmin and valueX < self.xmax:
            ind = next(x[0] for x in enumerate(self.right) if x[1] > valueX)
            self.y[ind]+=1

    def normalize(self):
        totN = float(sum(self.y))
        if not self.isLog:
            self.y = [yy/totN/self.dx for yy in self.y]
        else:
            self.y = [self.y[i]/totN/self.dx[i] for i in range(len(self.y))]

    def save(self,filepath):
        with open(filepath,'w') as saveFile:
            for i in range(len(self.x)):
                saveFile.write('%1.12f,%1.12f\n'%(self.x[i],self.y[i]))



def histCostVar(x,ndat,mindx):
    Cn = len(x)**-1
    x.sort()
    p = []
    dx = []
    mx = []

    ind1 = 0
    while ind1+ndat-1 < len(x): #-1?
        ind2 = ind1 + ndat-1
        ind3 = next(i for i in range(len(x)) if x[i] > x[ind2])

        dx.append(x[ind3]-x[ind1])
        while (dx[-1] < mindx):
            if (ind3 == len(x)-1):
                break
            ind3 = ind3+1
            dx[-1] = x[ind3]-x[ind1]
        #print indx,"\n"

        p.append((ind3-ind1)*Cn/float(dx[-1]))
        mx.append((x[ind3]+x[ind1])*0.5)
        ind1 = ind3

    return p,mx

def cumSumDist(x):
    xx = sorted(t_tot)
    Cn =  len(t_tot)**-1
    yy = [(y+1)*Cn for y in range(len(t_tot))]
    return yy,xx


def listCounts(aList):
    uni = set(aList)
    cnt = {}
    for u in uni:
        cnt[u]=0
    for x in aList:
        cnt[x]+=1
    return cnt


def runningAverage(t,field,numPt):
    indexSorted = sorted(range(len(t)), key=lambda k: t[k])
    x = []
    y = []
    for i in range(0,len(t)-numPt,numPt):
        indexSet = indexSorted[i:i+numPt]
        tM = 0
        fM = 0
        for j in indexSet:
            tM+=t[j]
            fM+=field[j]

        x.append(tM/float(numPt))
        y.append(fM/float(numPt))
    return x,y


def binAverage(t,field,dt):
    indexSorted = sorted(range(len(t)), key=lambda k: t[k])
    x = []
    y = []
    for i in range(0,len(t)):
        indexSet = indexSorted[i:i+numPt]
        tM = 0
        fM = 0
        for j in indexSet:
            tM+=t[j]
            fM+=field[j]
        
        x.append(tM/float(numPt))
        y.append(fM/float(numPt))
    return x,y

def runningAverageSmooth(t,field,numPt):
    indexSorted = sorted(range(len(t)), key=lambda k: t[k])
    x = []
    y = []
    for i in range(0,len(t)-numPt,numPt/10):
        indexSet = indexSorted[i:i+numPt]
        tM = 0
        fM = 0
        for j in indexSet:
            tM+=t[j]
            fM+=field[j]
        
        x.append(tM/float(numPt))
        y.append(fM/float(numPt))
    return x,y
"""


        
   