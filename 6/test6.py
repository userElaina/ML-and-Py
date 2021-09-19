import os
import csv
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import svm,tree
from sklearn.feature_selection import SelectFromModel,RFE
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV,LogisticRegression
def fLoadDataMatrix(filename) :
    fr = open(filename)
    Lines = fr.readlines()
    L1 = Lines[0].strip()
    tClass = L1.split('\t')[1:]
    Sample_cnt = len(tClass)
    tFeature = []
    tSample = list(range(Sample_cnt))
    tMatrix = np.zeros((Sample_cnt, len(Lines)-1))
    Feature_index = 0
    for line in Lines[1:] :
        line = line.strip()
        listfromline = line.split('\t')
        tFeature.append(listfromline[0])
        tMatrix[:, Feature_index] = listfromline[1:]
        Feature_index += 1
    return tSample, tClass, tFeature, tMatrix
def classify0(inX, Traindata, labels, k) :
    DataMatSize = Traindata.shape[0]
    diffMat = np.tile(inX, (DataMatSize, 1)) - Traindata
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k) :
        voteIlabel = labels[sortedDistIndices[i+1]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key = lambda d: d[1], reverse = True)
    return  sortedClassCount[0][0]
def trainNB(Traindata, labels) :
    numSample = len(Traindata)
    idxP = []
    idxN = []
    for i in range(numSample) :
        if (labels[i] == 1) : idxP.append(i)
        elif (labels[i] == 0) : idxN.append(i)
        else : pass
    Pmat = Traindata[idxP][:]
    Nmat = Traindata[idxN][:]
    pAbusive = len(idxP) / float(numSample)
    Pavg = np.mean(Pmat, 0) 
    Navg = np.mean(Nmat, 0) 
    Pdelta = np.var(Pmat, 0) 
    Ndelta = np.var(Nmat, 0)
    return Pavg, Navg, Pdelta, Ndelta, pAbusive
def Normaldistribution(mu, sig, x) :
    return math.exp(-math.pow(x - mu, 2)/(2*sig))/(math.sqrt(2*math.pi)*math.sqrt(sig))
def classifyNB(inX, Pavg, Navg, Pdelta, Ndelta, pAb) :
    numFeature = len(inX)
    p1 = pAb
    p0 = 1 - pAb
    for i in range(numFeature):
        p1 *=  Normaldistribution(Pavg[i], Pdelta[i], inX[i])
        p0 *=  Normaldistribution(Navg[i], Ndelta[i], inX[i])
    if (p1 > p0):
        return 1
    else:
        return 0
def Evaluate(predicted, origin) :
    ConfusionMat = np.zeros((2,3))
    for i in range(len(origin)):
        ConfusionMat[origin[i]][2] += 1
        ConfusionMat[origin[i]][predicted[i]] += 1
    Result = list(range(5))
    TP = ConfusionMat[1][1]
    TN = ConfusionMat[0][0]
    FP = ConfusionMat[1][0]
    FN = ConfusionMat[0][1]
    P = ConfusionMat[1][2]
    N = ConfusionMat[0][2]
    Result[0] = TP / P
    Result[1] = TN / N
    Result[2] = (TP + TN) / (P + N)
    Result[3] = (Result[0] + Result[1]) / 2
    x1 = (TP * TN - FP * FN)
    x2 = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    if (x2 == 0): Result[4] = 0
    else: Result[4] = x1 / math.sqrt(x2)
    return Result
def CrossValidation(DataMat, Labels, K) :
    idxP = []
    idxN = []
    for i in range(len(Labels)) :
        if (Labels[i] == 1) : idxP.append(i)
        elif (Labels[i] == 0) : idxN.append(i)
        else : pass
    random.shuffle(idxP)
    random.shuffle(idxN)
    GroupP = []
    GroupN = []
    for i in range(K): 
        GroupN.append(idxN[i::K])
        GroupP.append(idxP[i::K])
    
    result = np.zeros((5, 5))
    for Gcnt in range(K):
        Testdata = GroupN[Gcnt] + GroupP[Gcnt]
        Traindata = []
        for i in range(K):
            if (i != Gcnt) : Traindata = Traindata + GroupN[i] + GroupP[i]
            else: pass
        Trainx = DataMat[Traindata, :]
        Trainy = np.asarray([Labels[x] for x in Traindata])
        Testx = DataMat[Testdata, :]
        Testy = np.asarray([Labels[x] for x in Testdata])
        predict1 = np.zeros((len(Testy)), dtype = np.int)
        predict2 = np.zeros((len(Testy)), dtype = np.int)
        for i in range(len(Testy)) :
            predict1[i] = classify0(Testx[i], Trainx, Trainy, 3)
        Pavg, Navg, Pdelta, Ndelta, pAbusive = trainNB(Trainx, Trainy)
        for i in range(len(Testy)) :
            predict2[i] = classifyNB(Testx[i], Pavg, Navg, Pdelta, Ndelta, pAbusive)
        model1 = svm.SVC(kernel = 'linear')
        model2 = tree.DecisionTreeClassifier(criterion = 'entropy')
        model3 = LassoLarsCV(cv=5)
        model1.fit(Trainx, Trainy)
        model2.fit(Trainx, Trainy)
        model3.fit(Trainx, Trainy)
        predict3 = model1.predict(Testx)
        predict4 = model2.predict(Testx)
        predict5 = np.zeros((len(Testy)), dtype = np.int)
        r1 = Evaluate(predict1, Testy)
        r2 = Evaluate(predict2, Testy)
        r3 = Evaluate(predict3, Testy)
        r4 = Evaluate(predict4, Testy)
        predictarr = model3.predict(Testx)
        for i in range(len(Testy)) :
            if (predictarr[i] > 0.5): predict5[i] = 1
            else: predict5[i] = 0
        r5 = Evaluate(predict5, Testy)
        result +=  np.asarray([r1, r2, r3, r4, r5])
    return result / K
def T_Test(tClass, tMatrix) :
    idxP = []
    idxN = []
    for i in range(len(tClass)) :
        if (tClass[i] == 1) : idxP.append(i)
        elif (tClass[i] == 0) : idxN.append(i)
        else : pass
    (elTvalue, elPvalue) = stats.ttest_ind(tMatrix[idxP,:], tMatrix[idxN,:])
    idxRank = np.asarray(range(len(tMatrix[0])))
    elDict = {}
    for tI in range(len(idxRank)) : elDict[idxRank[tI]] =elPvalue[tI]
    tempRS = np.asarray(sorted(elDict.items(), key = lambda d: d[1])) [:,0]
    idxRankSort = np.asarray(range(len(tempRS)))
    for tI in range(len(tempRS)) : idxRankSort[tI] = int(tempRS[tI])
    return idxRankSort, elTvalue, elPvalue
def Histograms(Titles, ResultMat):
    Fig = plt.figure(figsize=(12, 7), dpi = 120)
    Xbase = np.asarray([1, 7, 13, 19, 25])
    for i in range(4):
        plt.subplot(221 + i)
        plt.ylim(-0.2, 1)
        plt.xlim(0, 35)
        plt.bar(Xbase, ResultMat[i][0], 1, label = "KNN", color = 'r')
        plt.bar(Xbase+1, ResultMat[i][1], 1, label = "NBayes", color = 'g')
        plt.bar(Xbase+2, ResultMat[i][2], 1, label = "SVM", color = 'b')
        plt.bar(Xbase+3, ResultMat[i][3], 1, label = "DTree", color = 'orange')
        plt.bar(Xbase+4, ResultMat[i][4], 1, label = "Lasso", color = 'purple')
        plt.xticks(Xbase+2, ('Sn', 'Sp', 'Acc', 'Avc', 'MCC'))
        plt.title(Titles[i])
        plt.legend(loc="upper right", fontsize = "small" )

fn=r'C:\Elaina\***\6\ALL3.txt'
Sample, Class, Feature, Matrix = fLoadDataMatrix(fn)
PNdict = {}
PNdict['NEG'] = 0
PNdict['POS'] = 1
Col = {0:'b', 1:'r'}
ClassLabels = [PNdict[x] for x in Class]
ColList = [Col[x] for x in ClassLabels]

idxRankSort, Tvalue, Pvalue = T_Test(ClassLabels, Matrix)
ttestFeatureidx = [idxRankSort[:1], idxRankSort[:10], idxRankSort[:100], idxRankSort[-100:]]
Re = []
for i in range(4):
    Re.append(CrossValidation(Matrix[:,ttestFeatureidx[i]], ClassLabels, 10))
title1 = ['Top-1', 'Top-10', 'Top-100', 'Bottom-100']
Histograms(title1, Re)

fo=r'C:\Elaina\***\6\ans.png'
plt.savefig(fo)