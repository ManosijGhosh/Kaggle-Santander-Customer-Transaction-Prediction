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
from confusion import evaluate
from knnClassifier import createKnnModel
from autoencoder import autoencoder
from sklearn import preprocessing
from neuralNetwork import neuralNetwork

'''
TO DO: DROP TIME ATTRIBUTE AND TRY
'''

def preprocess(choice=1):
	
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
		
def main():
	
	#X_label, X_train, Y_test = make_data()
	if not os.path.exists('models'):
		os.mkdir('models')
	data, labels = preprocess(2) # 1-kaggle, 2- credit card
	fold = 3
	count=0
	choice = 2 #0- knn_classifier, 1 - autoencoder, 2 - neural network
	str = 'test_log_mano_50'
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
			autoencoder(trainData, trainLabels, testData, str, count)
			maxAuc = maxThreshold = 0
			for i in range(0,20):
				thresh = i/10000
				autoencoderLabels = evaluate(('models/'+str+'_%i') %count, thresh)
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
				

main()

## THRESH 1.0: true fraud: 434, false fraud: 21397, total: 284800