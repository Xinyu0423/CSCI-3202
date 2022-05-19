from __future__ import print_function
import keras
import utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model

import os
from sklearn.model_selection import train_test_split

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from PIL import Image


def getVGGFeatures(fileList, layerName):
	# Initial Model Setup
	base_model = VGG16(weights='imagenet')
	model = Model(inputs=base_model.input, outputs=base_model.get_layer(layerName).output)

	# Confirm number of files passed is what was expected
	rArray = []
	print("Number of Files Passed:")
	print(len(fileList))

	for iPath in fileList:
		# Time Printing for Debug, you can comment this out if you wish
		now = datetime.now()
		current_time = now.strftime("%H:%M:%S")
		print("Current Time =", current_time)
		try:
			# Read Image
			img = image.load_img(iPath)
			# Update user as to which image is being processed
			print("Getting Features " + iPath)
			# Get image ready for VGG16
			img = img.resize((224, 224))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			# Generate Features
			internalFeatures = model.predict(x)
			rArray.append(internalFeatures.flatten())
		except:
			print("Failed " + iPath)
	return rArray

def cropImage(image, x1, y1, x2, y2):
	#utils.raiseNotDefined()
	box=(x1,y1,x2,y2)
	tempImage=image.crop(box)
	return tempImage



def standardizeImage(image, x, y):
	#utils.raiseNotDefined()
	tempImage=image.resize((x,y))
	return tempImage

def preProcessImages(images):
	#cropedImage=[]
	if not os.path.exists("Cropped Image"):
		os.mkdir("Cropped Image")
	for line in images:
		x1,y1,x2,y2=line.split()[5].split(",")
		#print(x1,y1,x2,y2)
		imageName=line.split()[7]
		#print(imageName)
		try:
			#if os.path.isfile("./Cropped Image/"+imageName):
			#	continue
			choppedImg=Image.open("./uncropped/"+imageName)
			tempImg=cropImage(choppedImg,int(x1),int(y1),int(x2),int(y2))
			tempImg=standardizeImage(tempImg,60,60)
			tempImg.save("./Cropped Image/"+imageName)
			#cropedImage.append(imageName)
		except OSError:
			print("Error!")

	#return cropedImage




def visualizeWeight():
	#You can change these parameters if you need to
	utils.raiseNotDefined()

def trainFaceClassifier(preProcessedImages, labels):
	#utils.raiseNotDefined()
	X_train, X_test, y_train, y_test = train_test_split(preProcessedImages,labels,test_size=0.2)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train, test_size=0.2)
	y_train = keras.utils.to_categorical(y_train, 6)
	y_test = keras.utils.to_categorical(y_test, 6)
	y_valid = keras.utils.to_categorical(y_valid, 6)

	model = Sequential()
	model.add(Dense(100, input_shape=(3600,)))
	model.add(Activation('relu'))

	#model.add(Dense(50, input_shape=(3600,)))
	#model.add(Activation('relu'))

	model.add(Dense(6))
	model.add(Activation('softmax'))
	#keras.layers.Dropout(0.1, noise_shape=None, seed=None)
	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

	history = model.fit(X_train, y_train,
						batch_size=20, epochs=100,
						verbose=2,
						validation_data=(X_valid, y_valid))

	score = model.evaluate(X_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	#get from https://keras.io/visualization/
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train Loss', 'Validation Loss'], loc='upper left')
	plt.show()
	return history


def trainFaceClassifier_VGG(extractedFeatures, labels):
	#utils.raiseNotDefined()
	#print(rArray[1].shape)
	#rArrayShape=rArray[1].shape
	#print(rArrayShape)
	X_train, X_test, y_train, y_test = train_test_split(extractedFeatures, labels, test_size=0.2)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)
	y_train = keras.utils.to_categorical(y_train, 6)
	y_test = keras.utils.to_categorical(y_test, 6)
	y_valid = keras.utils.to_categorical(y_valid, 6)

	model = Sequential()
	
	#model.add(Dense(100, input_shape=(25088,)))
	model.add(Dense(100, input_shape=(100352,)))
	model.add(Activation('relu'))

	model.add(Dense(6))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

	history = model.fit(X_train, y_train, batch_size=128, epochs=20, verbose=2, validation_data=(X_valid, y_valid))

	score = model.evaluate(X_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	return history


if __name__ == '__main__':
	print("Your Program Here")
	#---------------------------Part 1---------------------------------
	inputFile=open("downloadedFiles.txt")
	#preProcessImages(inputFile)

	#--------------------------Part 2-----------------------------------
	folder = os.fsencode("Cropped Image")
	fileNameList=[]
	for files in os.listdir(folder):
		filename = os.fsdecode(files)
		fileNameList.append(filename)
	fileNameList.sort()
	labelList=[]
	preProcessedImages = []
	for i in fileNameList:
		temp=i.split('.')
		if "bracco" in temp[0]:
			tempLabel=0
		elif "butler" in temp[0]:
			tempLabel=1
		elif "gilpin" in temp[0]:
			tempLabel=2
		elif "harmon" in temp[0]:
			tempLabel=3
		elif "radcliffe" in temp[0]:
			tempLabel=4
		elif "vartan" in temp[0]:
			tempLabel=5
		labelList.append(tempLabel)

		tempImage = Image.open("./Cropped Image/" + i).convert('L')
		tempData = tempImage.getdata()
		tempPPImages = np.array(tempData) / 225
		# print(tempPPImages)
		preProcessedImages.append(tempPPImages)


	preProcessedImages=np.array(preProcessedImages)
	labelList=np.array(labelList)
	#print(len(preProcessedImages[1]))
	#print(preProcessedImages.shape)
	#print(labelList.shape)
	history=trainFaceClassifier(preProcessedImages,labelList)

	#--------------------------------Part3----------------------------
	fileList=[]
	for i in fileNameList:
		temp="./Cropped Image/" + i
		fileList.append(temp)
	rArray=getVGGFeatures(fileList,"block4_pool")
	rArray=np.array(rArray)
	VGGHistory=trainFaceClassifier_VGG(rArray,labelList)


