'''
AUTHOR: SOUMYADEEP THAKUR
DATE: 6 OCT 2018
'''

import os
import numpy as np
import signal
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

from confusion import evaluate
from autoencoder import autoencoder
from neuralNetwork import neuralNetwork
from boltzmannMachine import restrictedBoltzmannMachine
from classifiers import createKnnModel
from classifiers import randomForest
from classifiers import featureSelectionMI

'''
TO DO: DROP TIME ATTRIBUTE AND TRY
'''

def preprocess():
	
	df = pd.read_csv('Data/train.csv')
	data = df
	data = data.drop(['ID_code'],axis=1)

	# Seperate X and Y data

	Y_val = data['target']
	X_val = data.drop(['target'], axis=1)
	X_val = preprocessing.normalize(X_val.values)
	Y_val = Y_val.values

	'''
	from sklearn.preprocessing import StandardScaler

	data['normAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))
	'''
	return X_val, Y_val

def split(X_val, Y_val,fold):

    # Prepare for stratified KFold crossvalidation

	skf = StratifiedKFold(n_splits=fold)
	skf.get_n_splits(X_val, Y_val)
	#print(skf)

	for train_index, test_index in skf.split(X_val, Y_val):
		yield (train_index, test_index)

def consfusion_eval(actualLabels,predictedLabels):

	#sprint(labels)
	#TPX, TNX, FPX, FNX = list(), list(), list(), list()
	
	#print(len(actualLabels),' ',len(predictedLabels))
	
	
	if (len(actualLabels)!=len(predictedLabels)):
		print ('error final label not same as actual')
	
	tp = tn = fp = fn = 0
	for j in range(len(predictedLabels)):
		if (predictedLabels[j]==actualLabels[j]):
			if actualLabels[j]==0:
				tp+=1
			else:
				tn+=1
		else:
			if actualLabels[j]==1:
				fp+=1
			else:
				fn+=1;
	print('Conf: ', tp, ' -- ', tn, ' -- ', fp, ' -- ', fn),
	auc = (tp/(2*(tp+fn))) + (tn/(2*(fp+tn)))
	print(' AUC - ', auc)
	return auc
		
def classifier():
	
	#X_label, X_train, Y_test = make_data()
	if not os.path.exists('models'):
		os.mkdir('models')
	data, labels = preprocess()
	fold = 3
	count=0
	choice = 5
	#0- knn_classifier, 1 - autoencoder, 2 - neural network, 3 - bernoilli restricted boltzmann machine
	fileName = 'test_log_mano_50'
	for train_index, test_index in split(data, labels, fold):
		count += 1
		trainData,testData = data[train_index], data[test_index]
		trainLabels, testLabels = labels[train_index], labels[test_index]

		print('splitting done')
		print('Number of 1 - ', sum(testLabels), ' 0 - ',(len(testLabels)-sum(testLabels)))

		if (choice == 0):
			knnLabels = createKnnModel(trainData, trainLabels, testData)
			print('knn trained')
			auc = consfusion_eval(testLabels, knnLabels)
			print('AUC - ',auc)
		elif (choice == 1):
			autoencoder(trainData, trainLabels, testData, fileName, count)
			maxAuc = maxThreshold = 0
			for i in range(0,20):
				thresh = i/10000
				autoencoderLabels = evaluate(('models/'+fileName+'_%i') %count, thresh)
				print('Thresh - ', thresh),
				temp = consfusion_eval(testLabels, autoencoderLabels)
				if (temp>maxAuc):
					maxAuc = temp
					maxThreshold = thresh
			print('Max auc - ',maxAuc,' threshold - ',maxThreshold)
		elif (choice == 2):
			nnLabels = neuralNetwork(trainData, trainLabels, testData, count)
			auc = consfusion_eval(testLabels, nnLabels)
			print('for fold - ', count,' AUC - ',auc)
		elif (choice == 3):
			boltzLabels = restrictedBoltzmannMachine(trainData, trainLabels, testData)
			auc = consfusion_eval(testLabels, boltzLabels)
			print('AUC - ',auc)
		elif (choice == 4):
			rfLabels = randomForest(trainData, trainLabels, testData)
			auc = consfusion_eval(testLabels, rfLabels)
			print('AUC - ',auc)
		elif (choice==5):
			featureSelectionMI(trainData, trainLabels, testData, testLabels)
				
def generateResult():
	if not os.path.exists('models'):
		os.mkdir('models')
	trainData, trainLabels = preprocess() # 1-kaggle, 2- credit card
	
	# get test data
	testData = pd.read_csv('Data/test.csv')
	testData = testData.drop(['ID_code'],axis=1)
	testData = preprocessing.normalize(testData.values)
	
	
	#testLabels = neuralNetwork(trainData, trainLabels, testData, 0)
	testLabels = featureSelectionMI(trainData, trainLabels, testData)

	print('Predicted number of 1 - ',format(sum(testLabels==1)))
	print('Predicted number of 0 - ',format(sum(testLabels==0)))

	file = open('sample_submission.csv','w') 
 	
	file.write('ID_code,target\n') 
	for i in range(0,testLabels.shape[0]):
		string = 'test_'+str(i)+','+str(int(testLabels[i]))+'\n'
		file.write(string)
			
#classifier()
generateResult()

## THRESH 1.0: true fraud: 434, false fraud: 21397, total: 284800