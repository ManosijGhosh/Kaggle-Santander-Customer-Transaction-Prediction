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
from autoencoder_kfold import autoencoder
from sklearn import preprocessing

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
	
	print(len(actualLabels),' ',len(predictedLabels))
	
	'''
	if (len(actualLabels)!=len(predictedLabels)):
		print ('error final label not same as actual')
	'''
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
	print('Conf: ', tp, ' -- ', tn, ' -- ', fp, ' -- ', fn)
		
def main():
	
	#X_test, X_train, Y_test = make_data()
	if not os.path.exists('models'):
		os.mkdir('models')
	X_val, Y_val = preprocess()
	fold = 3
	count=0
	str = 'test_log_mano_50'
	for train_index, test_index in split(X_val, Y_val, fold):
		count += 1
		X_train, X_test = X_val[train_index], X_val[test_index]
		Y_train, Y_test = Y_val[train_index], Y_val[test_index]

		print('splitting done')
		print('Number of 1 - ', sum(Y_test), ' 0 - ',(len(Y_test)-sum(Y_test)))
		'''
		knnLabels = createKnnModel(X_train, X_test, Y_train)
		print('knn trained')
		consfusion_eval(Y_test, knnLabels)
		
		'''

		autoencoder(X_train, X_test, Y_train, str, count)
		for i in range(0,10):
			autoencoderLabels = evaluate(('models/'+str+'_%i') %count, (i/100))
			print('Thresh - ', (i/100)),
			consfusion_eval(Y_test, autoencoderLabels)
		#'''
		

main()

## THRESH 1.0: true fraud: 434, false fraud: 21397, total: 284800