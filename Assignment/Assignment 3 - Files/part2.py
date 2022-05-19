import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import OrthogonalMatchingPursuit

#import csv


#Label = "Credit"
Label = "0"
Features = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16","A17","A18","A19"]

def saveBestModel(clf):
    pickle.dump(clf, open("bestModel.model", 'wb'))

def readData(file):
    df = pd.read_csv(file)
    return df

np.set_printoptions(suppress=True)

def trainOnAllData(df,clf):
    x = []
    y = []
    for i in range(0, len(df)):
        tempX = df.loc[i, Features].tolist()
        tempY = -1
        if df.loc[i, "Credit"] == "good":
            tempY = 1
        elif df.loc[i, "Credit"] == "bad":
            tempY = 0
        x.append(tempX)
        y.append(tempY)
    newX = np.array(x)
    newY = np.array(y)
    clf.fit(newX,newY)
    saveBestModel(clf)


def trainData(df, clf):
    #Use this function for part 4, once you have selected the best model
    #print("TODO")
    x=[]
    y=[]
    for i in range(0,len(df)):
        tempX=df.loc[i,Features].tolist()
        tempY=-1
        if df.loc[i,"Credit"]=="good":
            tempY=1
        elif df.loc[i,"Credit"]=="bad":
            tempY=0
        x.append(tempX)
        y.append(tempY)
    print(x)
    #print(y)
    kf = KFold(n_splits=10)
    AUROCList=[]
    for trainIndex,testIndex in kf.split(x,y):
        trainX,trainY=np.array(x)[trainIndex],np.array(y)[trainIndex]
        testX,testY=np.array(x)[testIndex],np.array(y)[testIndex]
        #print(trainIndex)
        #print("---------------------------------------------------")
        #print(testIndex)
        clf.fit(trainX,trainY)

        y_pred=clf.predict(testX)
        #print(y_pred)
        AUROC=roc_auc_score(testY,y_pred)
        #AUROCList.np.append(AUROC)
        AUROCList.append(AUROC)

    NPAUROCList=np.array(AUROCList)
    #print(AUROCList)
    avg=np.average(NPAUROCList)
    stdDev=np.std(NPAUROCList)
    print("The average of AUROC is:",round(avg, 4))
    print("The stand deviation of AUROC is:",round(stdDev, 4))
    return avg

    #print(testX)

def bestModel(clf):
    x = []
    y = []
    tp=0
    fp=0
    tn=0
    fn=0
    for i in range(0, len(df)):
        tempX = df.loc[i, Features].tolist()
        print(tempX)
        tempY = -1
        if df.loc[i, "Credit"] == "good":
            tempY = 1
        elif df.loc[i, "Credit"] == "bad":
            tempY = 0
        x.append(tempX)
        y.append(tempY)
    newX=np.array(x)
    newY=np.array(y)
    clf.fit(newX,newY)
    y_pred = clf.predict(newX)
    AUROC = roc_auc_score(y,y_pred)
    for i in range(0,len(df)):
        if y_pred[i]==1 :
            if y_pred[i]==y[i]:
                tp=tp+1
            else:
                fp=fp+1
        elif y_pred[i]==0:
            if y_pred[i] == y[i]:
                tn=tn+1
            else:
                fn=fn+1
    '''for i in range(0,len(df)):
        if y_pred[i]==1 and y[i]==1:
            tp=tp+1
        elif y_pred[i]==1 and y[i]==0:
            fp=fp+1
        elif y_pred[i]==0 and y[i]==0:
            tn=tn+1
        elif y_pred[i]==0 and y[i]==1:
            fn=fn+1'''

    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    accuracy=(tn+tn)/(tn+fp+tn+fn)
    print("The tp value is", tp)
    print("The fp value is", fp)
    print("The tn value is", tn)
    print("The fn value is", fn)
    print("The accuracy value is", round(accuracy,4))
    print("The precision value is",round(precision,4))
    print("The recall value is",round(recall,4))
    print("The AUROC value is",round(AUROC,4))

    creditList=[]
    for i in range(0,len(df)):
        temp=[]
        if newY[i]==1:
            temp.append("good")
        elif newY[i]==0:
            temp.append("bad")
        if y_pred[i]==1:
            temp.append("good")
        elif y_pred[i]==0:
            temp.append("bad")
        creditList.append(temp)
    #print(creditList)
    data=pd.DataFrame(x,columns=Features)
    data["Credit"]=[i[0] for i in creditList]
    data["Prediction"] = [i[1] for i in creditList]
    data.to_csv("bestModel.output",index=False)





df = readData("credit_train.csv")
#print(type(df))
'''trainData(df,LogisticRegression())
trainData(df,GaussianNB())
trainData(df,svm.SVC())
trainData(df,tree.DecisionTreeClassifier())
trainData(df,RandomForestClassifier())
trainData(df,MLPClassifier())
trainData(df,OrthogonalMatchingPursuit())

bestModel(LogisticRegression())
trainOnAllData(df,LogisticRegression())'''
trainData(df,LogisticRegression())



'''SCVList=[]
SCVList.append(trainData(df,svm.SVC(C=0.1,gamma=1)))
SCVList.append(trainData(df,svm.SVC(C=0.1,gamma=0.1)))
SCVList.append(trainData(df,svm.SVC(C=0.1,gamma=0.001)))
SCVList.append(trainData(df,svm.SVC(C=0.1,gamma=0.00001)))

SCVList.append(trainData(df,svm.SVC(C=1,gamma=1)))
SCVList.append(trainData(df,svm.SVC(C=1,gamma=0.1)))
SCVList.append(trainData(df,svm.SVC(C=1,gamma=0.001)))
SCVList.append(trainData(df,svm.SVC(C=0.1,gamma=0.00001)))

SCVList.append(trainData(df,svm.SVC(C=10,gamma=1)))
SCVList.append(trainData(df,svm.SVC(C=10,gamma=0.1)))
SCVList.append(trainData(df,svm.SVC(C=10,gamma=0.001)))
SCVList.append(trainData(df,svm.SVC(C=10.1,gamma=0.00001)))

SCVList.append(trainData(df,svm.SVC(C=100,gamma=1)))
SCVList.append(trainData(df,svm.SVC(C=100,gamma=0.1)))
SCVList.append(trainData(df,svm.SVC(C=100,gamma=0.001)))
SCVList.append(trainData(df,svm.SVC(C=100,gamma=0.00001)))

SCVList.append(trainData(df,svm.SVC(C=1000,gamma=1)))
SCVList.append(trainData(df,svm.SVC(C=1000,gamma=0.1)))
SCVList.append(trainData(df,svm.SVC(C=1000,gamma=0.001)))
SCVList.append(trainData(df,svm.SVC(C=1000,gamma=0.00001)))

print(max(SCVList))
print(SCVList.index(max(SCVList)))'''

'''RForestList=[]

RForestList.append(trainData(df,RandomForestClassifier(n_estimators=20, max_depth=2)))
RForestList.append(trainData(df,RandomForestClassifier(n_estimators=20, max_depth=4)))
RForestList.append(trainData(df,RandomForestClassifier(n_estimators=20, max_depth=6)))
RForestList.append(trainData(df,RandomForestClassifier(n_estimators=20, max_depth=8)))

RForestList.append(trainData(df,RandomForestClassifier(n_estimators=40, max_depth=2)))
RForestList.append(trainData(df,RandomForestClassifier(n_estimators=40, max_depth=4)))
RForestList.append(trainData(df,RandomForestClassifier(n_estimators=40, max_depth=6)))
RForestList.append(trainData(df,RandomForestClassifier(n_estimators=40, max_depth=8)))

RForestList.append(trainData(df,RandomForestClassifier(n_estimators=60, max_depth=2)))
RForestList.append(trainData(df,RandomForestClassifier(n_estimators=60, max_depth=4)))
RForestList.append(trainData(df,RandomForestClassifier(n_estimators=60, max_depth=6)))
RForestList.append(trainData(df,RandomForestClassifier(n_estimators=60, max_depth=8)))

RForestList.append(trainData(df,RandomForestClassifier(n_estimators=80, max_depth=2)))
RForestList.append(trainData(df,RandomForestClassifier(n_estimators=80, max_depth=4)))
RForestList.append(trainData(df,RandomForestClassifier(n_estimators=80, max_depth=6)))
RForestList.append(trainData(df,RandomForestClassifier(n_estimators=80, max_depth=8)))

RForestList.append(trainData(df,RandomForestClassifier(n_estimators=100, max_depth=2)))
RForestList.append(trainData(df,RandomForestClassifier(n_estimators=100, max_depth=4)))
RForestList.append(trainData(df,RandomForestClassifier(n_estimators=100, max_depth=6)))
RForestList.append(trainData(df,RandomForestClassifier(n_estimators=100, max_depth=8)))

print(max(RForestList))
print(RForestList.index(max(RForestList)))'''

'''LogisticRList=[]
LogisticRList.append(trainData(df,LogisticRegression(penalty='l1',max_iter=200)))
LogisticRList.append(trainData(df,LogisticRegression(penalty='l1',max_iter=400)))
LogisticRList.append(trainData(df,LogisticRegression(penalty='l1',max_iter=600)))
LogisticRList.append(trainData(df,LogisticRegression(penalty='l1',max_iter=800)))
LogisticRList.append(trainData(df,LogisticRegression(penalty='l1',max_iter=1000)))

LogisticRList.append(trainData(df,LogisticRegression(penalty='l2',max_iter=200)))
LogisticRList.append(trainData(df,LogisticRegression(penalty='l2',max_iter=400)))
LogisticRList.append(trainData(df,LogisticRegression(penalty='l2',max_iter=600)))
LogisticRList.append(trainData(df,LogisticRegression(penalty='l2',max_iter=800)))
LogisticRList.append(trainData(df,LogisticRegression(penalty='l2',max_iter=1000)))
print(max(LogisticRList))
print(LogisticRList.index(max(LogisticRList)))'''



