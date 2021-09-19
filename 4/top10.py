import os
import csv
import numpy as np
import pandas as pd
from scipy import stats

nclean=False
sclean=''
def Dreader(tn):
    global nclean
    global sclean
    nclean=False
    if tn[-4:len(tn)]=='xlsx':
        try:
            data_xls=pd.read_excel(tn,index_col=0)
        except FileNotFoundError:
            nclean=False
            return 'Error: This file does not exist.'
        tn=tn[0:-4]+'csv'
        nclean=True
        sclean=tn
        data_xls.to_csv(tn)
    if tn[-3:len(tn)]=='xls':
        try:
            data_xls=pd.read_excel(tn,index_col=0)
        except FileNotFoundError:
            nclean=False
            return 'Error: This file does not exist.'
        tn=tn[0:-3]+'csv'
        nclean=True
        sclean=tn
        data_xls.to_csv(tn)
    if tn[-3:len(tn)]=='txt':
        try:
            td=open(tn,'r')
        except FileNotFoundError:
            nclean=False
            return 'Error: This file does not exist.'
        td=td.read().split('\n')
        for i in range(len(td)):
            td[i]=td[i].split('\t')
        n=len(td)-1
        l=len(td[0])
        ts=list(range(l-1))
        for i in range(1,l):
            ts[i-1]=i-1
        tc=td[0][1:]
        tm=list(range(n-1))
        tf=list(range(n-1))
        for i in range(1,n):
            tf[i-1]=td[i][0];
            tm[i-1]=[float(x)for x in td[i][1:]]
        tm=np.transpose(tm)
    else:
        try:
            td=csv.reader(open(tn,'r'))
        except FileNotFoundError:
            nclean=False
            return 'Error: This file does not exist.'
        except Exception:
            nclean=False
            return 'Error: This function can read excel and csv only.'
        td=list(td)
        n=len(td)
        m=1
        for i in range(len(td[0])):
            if td[0][i]=='Class' or td[0][i]=='class':
                m=i
                break
        tf=td[0][1:m]+td[0][m+1:]
        tc=list(range(n-1))
        ts=list(range(n-1))
        tm=list(range(n-1))
        for i in range(1,n):
            ts[i-1]=td[i][0]
            tc[i-1]=td[i][m]
            tm[i-1]=[float(x)for x in td[i][1:m]]+[float(x)for x in td[i][m+1:]]
    return ts,tc,tf,tm

fn=r'C:\Elaina\***\4\ALL3.txt'
S,C,F,M=Dreader(fn)
idxP=[]
idxN=[]
for i in range(len(C)):
    if C[i]=='NEG':
        idxN.append(i)
    else:
        idxP.append(i)

(elTvalue,elPvalue)=stats.ttest_ind(M[idxP,:],M[idxN,:])
idxRank=np.asarray(range(len(M[0])))
elDict={}
for tI in range(len(idxRank)):elDict[idxRank[tI]]=elPvalue[tI]
tempRS=np.asarray(sorted(elDict.items(),key=lambda d:d[1]))[:,0]
idxRankSort=np.asarray(range(len(tempRS)))
for tI in range(len(tempRS)):idxRankSort[tI]=int(tempRS[tI])

n=10
for i in range(n):
    print(F[idxRankSort[i]],end=' ')