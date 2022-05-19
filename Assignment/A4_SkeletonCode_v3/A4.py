import numpy as np
import utils
import pandas as pd
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

#Perfect Instances
five =  [0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0]
two = [0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0]
patterns = [five,two]

def loadGeneratedData():
	df = pd.read_csv("xiji6874-TrainingData.csv")
	return df


def distort_input(instance, percent_distortion):

    #percent distortion should be a float from 0-1
    #Should return a distorted version of the instance, relative to distoriton Rate
    #print("TODO")
    #utils.raiseNotDefined()
	newInstance=[]
	for i in range(0,len(instance)):
		tempRand = np.random.random()
		if percent_distortion>tempRand:
			if instance[i]==1:
				newInstance.append(0)
			elif instance[i]==0:
				newInstance.append(1)
		else:
			newInstance.append(instance[i])
	return newInstance

class HopfieldNetwork:
	def __init__(self, size):
		self.h = np.zeros([size,size])
		self.temp = np.zeros([size, size])

	def addSinglePattern(self, p):
		#Update the hopfield matrix using the passed pattern
		#print("TODO")
		#utils.raiseNotDefined()
		for i in range(0,len(self.h)):
			for j in range(0,len(self.h[0])):
				if i==j:
					continue
				temp=(2*p[i]-1)*(2*p[j]-1)
				#print(temp)
				self.h[i][j]+=temp
		# return self.h


	def fit(self, patterns):
		# for each pattern
		# Use your addSinglePattern function to learn the final h matrix
		#print("TODO")
		#utils.raiseNotDefined()
		'''temp = np.zeros([len(self.h), len(self.h)])
		for i in range(0, len(patterns)):
			self.addSinglePattern(patterns[i])
			temp = temp + self.h
		self.h = temp
		#print(self.h)'''

		for i in range(0,len(patterns)):
			self.addSinglePattern(patterns[i])
		# self.h=self.temp
		# return self.h

	def retrieve(self, inputPattern):
		#Use your trained hopfield network to retrieve and return a pattern based on the
		#input pattern.
		#HopfieldNotes.pdf on canvas is a good reference for asynchronous updating which
		#has generally better convergence properties than synchronous updating.
		#print("TODO")
		#utils.raiseNotDefined()

		#orderList=[2,0,4,1,3]
		tempResult=0
		changeList=[]
		while len(changeList)!=25:
			changeList=[]
			orderList=np.random.choice(range(25), 25, replace=False)
			for i in orderList:
				tempResult=0

				for j in range(0,len(self.h[i])):
					tempResult=self.h[i][j]*inputPattern[j]+tempResult

				if tempResult>=0:
					if inputPattern[i]==1:
						changeList.append(0)
					inputPattern[i]=1
				elif tempResult<0:
					if inputPattern[i]==0:
						changeList.append(0)
					inputPattern[i]=0
		#print(inputPattern)
		return inputPattern

	def classify(self, inputPattern):
		#Classify should consider the input and classify as either, five or two
		#You will call your retrieve function passing the input
		#Compare the returned pattern to the 'perfect' instances
		#return a string classification 'five', 'two' or 'unknown'
		#print("TODO")
		#utils.raiseNotDefined()
		resultPattern=self.retrieve(inputPattern)
		if resultPattern==two:
			return "two"
		elif resultPattern==five:
			return "five"
		else:
			return "unknown"

	def MLP(self,inputPattern):
		x = patterns
		y=['five','two']
		clf = MLPClassifier()
		clf.fit(x, y)
		MLPResult=clf.predict(inputPattern)
		return MLPResult

if __name__ == "__main__":
	hopfieldNet = HopfieldNetwork(25)

	#utils.visualize(five)
	#utils.visualize(two)


	#hopfieldNet.fit(patterns)
	#addStrList=hopfieldNet.addSinglePattern(five)
	#print(addStrList)
	#fitList=hopfieldNet.fit(patterns)
	#print(fitList)
	df = loadGeneratedData()
	Features=['r00','r01','r02','r03','r04','r10','r11','r12','r13','r14','r20','r21','r22','r23','r24','r30','r31','r32','r33','r34','r40','r41','r42','r43','r44']
	myList=[]
	for i in range(0, len(df)):
		tempList = df.loc[i, Features].tolist()
		myList.append(tempList)

#---------------------------------------Part2-----------------------------------------------
	hopfieldNet.fit(patterns)
	#print(hopfieldNet.h)
	clfList=[]
	labelList=[]
	countACC=0
	for i in myList:
		clfList.append(hopfieldNet.classify(i))
	#print(clfList)
	for i in range(0,len(df)):
		labelList.append(df.loc[i, "label"])

	for i in range(0,len(labelList)):
		if labelList[i]==clfList[i]:
			countACC=countACC+1

	accuracy=countACC/len(labelList)
	#print("The accracy of Hopfield is ",accuracy)
	#print(clfList)

	# ---------------------------------------Part3-----------------------------------------------
	MLPResult=hopfieldNet.MLP(myList)
	#print(MLPResult)
	MLPCount=0
	for i in range(0,len(MLPResult)):
		if labelList[i]==MLPResult[i]:
			MLPCount=MLPCount+1
	MLPAccuracy=MLPCount/len(labelList)
	#print("The accuracy of MLP is",MLPAccuracy)

	# ---------------------------------------Part4-----------------------------------------------
	distortList=distort_input(myList[1], 0.1)
	#print(myList[1])
	#print(distortList)
	distRates=0.0
	percentDistList=[]
	while distRates<0.5:
		percentDistList.append(round(distRates, 2))
		distRates=distRates+0.01

	accuracyHop = []
	accuracyMLP=[]
	for i in range(0,len(percentDistList)):
		distHopList=[]
		distMLPList=[]
		countHop=0
		countMLP=0
		for j in range(0,len(myList)):
			temp=distort_input(myList[j],percentDistList[i])
			#print("temp!!!!!!!",temp)
			tempCLFResult=hopfieldNet.classify(temp)
			tempMLPResult=hopfieldNet.MLP([temp])
			distHopList.append(tempCLFResult)
			distMLPList.append(tempMLPResult)

		for k in range(0,len(distHopList)):
			if distHopList[k]==labelList[k]:
				countHop=countHop+1
			if distMLPList[k]==labelList[k]:
				countMLP=countMLP+1

		tempAccHop=countHop/len(labelList)
		accuracyHop.append(tempAccHop)
		tempAccMLP=countMLP/len(labelList)
		accuracyMLP.append(tempAccMLP)

	plt.plot(accuracyHop,color='skyblue', label="Hopfield Network")
	plt.plot(accuracyMLP, color='red', label="MLP")
	#plt.xlim((0, 0.5))
	#plt.xticks(np.arange(0, 0.5, 0.1))
	plt.xticks([])
	plt.xlabel("Distort Rate")
	plt.ylabel("Accuracy")
	plt.legend(loc='best')
	plt.show()

# ---------------------------------------Part5-----------------------------------------------
	df2=pd.read_csv("NewInput.csv")
	totalList=myList
	totalLabel=labelList
	for i in range(0,len(df2)):
		tempList = df2.loc[i, Features].tolist()
		totalList.append(tempList)

	for i in range(0,len(df2)):
		labelList.append(df2.loc[i, "label"])

	percentDistList = []
	distRates = 0.0
	while distRates < 0.5:
		percentDistList.append(round(distRates, 2))
		distRates = distRates + 0.01

	x = totalList
	y = totalLabel
	#print(x)
	#print(y)
	accuracyMLP1 = []
	accuracyMLP2 = []
	accuracyMLP3 = []
	for i in range(0, len(percentDistList)):
		distMLPList1 = []
		distMLPList2 = []
		distMLPList3 = []
		countMLP1 = 0
		countMLP2 = 0
		countMLP3 = 0
		for j in range(0, len(totalList)):
			temp = distort_input(totalList[j], percentDistList[i])

			clf1 = MLPClassifier((10,5,1))
			clf1.fit(x, y)
			MLPResult1 = clf1.predict([temp])
			distMLPList1.append(MLPResult1)

			clf2 = MLPClassifier((100, 50, 10))
			clf2.fit(x, y)
			MLPResult2 = clf2.predict([temp])
			distMLPList2.append(MLPResult2)

			clf3 = MLPClassifier((1000, 500, 100))
			clf3.fit(x, y)
			MLPResult3 = clf3.predict([temp])
			distMLPList3.append(MLPResult3)


		for k in range(0, len(distMLPList1)):
			if distMLPList1[k] == totalLabel[k]:
				countMLP1 = countMLP1 + 1
			if distMLPList2[k] == totalLabel[k]:
				countMLP2 = countMLP2 + 1
			if distMLPList3[k] == totalLabel[k]:
				countMLP3 = countMLP3 + 1

		tempAccMLP1 = countMLP1 / len(totalLabel)
		accuracyMLP1.append(tempAccMLP1)
		tempAccMLP2 = countMLP2 / len(totalLabel)
		accuracyMLP2.append(tempAccMLP2)
		tempAccMLP3 = countMLP3 / len(totalLabel)
		accuracyMLP3.append(tempAccMLP3)


	plt.plot(accuracyMLP1, color='red', label="First MLP")
	plt.plot(accuracyMLP2, color='blue', label="Second MLP")
	plt.plot(accuracyMLP3, color='yellow', label="Third MLP")
	# plt.xlim((0, 0.5))
	# plt.xticks(np.arange(0, 0.5, 0.1))
	plt.xticks([])
	plt.xlabel("Distort Rate")
	plt.ylabel("Accuracy")
	plt.legend(loc='best')
	plt.show()





