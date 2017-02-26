from numpy import *
import math
import matplotlib.pyplot as plt

data = load("/Users/deepaliKamat/Downloads/data-2.npy")


def KNN(data,k):
    x1= data[:,0]
    x2=data[:,1]
    Y = data[:, 2]
    Y_pred = []

    tp=0
    tn=0
    fp=0
    fn=0
    for i in range(len(Y)):
        dist=[]
        for j in range(len(Y)):
            dist.append(math.sqrt(((x1[i]-x1[j])**2)+ (x2[i]-x2[j])**2))

        argDist = argsort(dist)
        arg2 = argDist[: k]
        sum1 = 0
        for i in arg2:
            sum1 = sum1+ Y[i]

        Y1_pre = sum1/k
        if Y1_pre > 0.5:
            Y_pred.append(1)
        else:
            Y_pred.append(0)

    for i in range(200):
        if (Y_pred[i]==0) and (Y[i]==0):
            tp=tp+1
        elif (Y_pred[i]==0) and (Y[i]==1):
            fp=fp+1
        elif (Y_pred[i]==1) and (Y[i]==0):
            fn=fn+1
        else:
            tn=tn+1
    print tp
    print tn
    print fp
    print fn



KNN(data,15)









