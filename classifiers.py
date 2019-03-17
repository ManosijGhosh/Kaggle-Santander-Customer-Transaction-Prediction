import numpy as np
from random import *

from sklearn import linear_model, datasets, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import pearsonr
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

from neuralNetwork import neuralNetwork


#repeated here from main
def consfusion_eval(actualLabels,predictedLabels):

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

def sampling(trainData, trainLabels, isSmoteDone, isUnderSamplingDone):
	if(isSmoteDone == 1 or isUnderSamplingDone == 1):
		print("Before OverSampling, counts of label '1': {}".format(sum(trainLabels==1)))
		print("Before OverSampling, counts of label '0': {} \n".format(sum(trainLabels==0)))
		if (isSmoteDone==1):
			sm = SMOTE(random_state = 2, k_neighbors = 4, n_jobs = -1)
			trainData, trainLabels = sm.fit_sample(trainData, trainLabels.ravel())

		elif (isUnderSamplingDone == 1):
			negatives = list()
			for labels in range(0,trainData.shape[0]):
				if ((trainLabels[labels]==0) and (random() < 0.2)):
					negatives.append(labels)
			print(len(negatives))
			trainData = np.delete(trainData, negatives, axis=0)
			trainLabels = np.delete(trainLabels, negatives)

		print("After OverSampling, counts of label '1': {}".format(sum(trainLabels==1)))
		print("After OverSampling, counts of label '0': {}".format(sum(trainLabels==0)))
	return trainData, trainLabels

def logisticRegression(trainData, trainLabels, testData):
	logistic = linear_model.LogisticRegression(solver='lbfgs', verbose=True, max_iter=5000, class_weight={0:1,1:10})
	
	trainData, trainLabels = sampling(trainData, trainLabels, 0, 1)
	logistic.C = 100
	logistic.fit(trainData, trainLabels)
	labels = logistic.predict(testData)
	#labels = list(labels)
	return labels

def createKnnModel(trainData, trainLabels, testData):
	knnModel = KNeighborsClassifier(n_neighbors=3, n_jobs = 4)
	knnModel.fit(trainData, trainLabels)
	print('knnTrained')
	results = knnModel.predict(testData)
	return results

def randomForest(trainData, trainLabels, testData):
	randomForest = RandomForestClassifier(n_estimators = 100,n_jobs = 4, verbose = True, class_weight={0:1,1:10}, criterion = 'entropy')
	randomForest.fit(trainData, trainLabels)
	print('random forest trained')
	labels = randomForest.predict(testData)
	return labels

def featureSelectionMI(trainData, trainLabels, testData):
	# gives horrible results
	n = trainData.shape[1]
	print(n)
	miValue=np.repeat(0.0,n)
	for i in range(0,n-1):
		#miValue[i] = metrics.normalized_mutual_info_score(trainData[:,i],trainLabels, average_method='geometric')
		#miValue[i] = metrics.mutual_info_score(trainData[:,i],trainLabels)
		miValue[i] , p_value = pearsonr(trainData[:,i],trainLabels)
	print(miValue)
	ranking=(-miValue).argsort()
	print(ranking)

	'''
	for loopval in range(10,199,20):
		temp=ranking[0:loopval]
		#print(temp)
		traindatanew=trainData[:,temp]
		testdatanew=testData[:,temp]
		labels = logisticRegression(traindatanew, trainLabels, testdatanew)
		auc = consfusion_eval(testLabels, labels)
		print('for ',loopval,' features; AUC - ',auc)
	'''
	temp=ranking[0:169]
	#print(temp)
	traindatanew=trainData[:,temp]
	testdatanew=testData[:,temp]
	labels = logisticRegression(traindatanew, trainLabels, testdatanew)
	return labels

def svmClassifier(trainData, trainLabels, testData):
	isSmoteDone = 0
	isUnderSamplingDone = 0

	trainData, trainLabels = sampling(trainData, trainLabels, isSmoteDone, isUnderSamplingDone)

	model = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, C=1.0, 
		fit_intercept=True, intercept_scaling=1, class_weight={0:1,1:9}, verbose=True, 
		random_state=None, max_iter=2000)
	# penalty - l1, l2; loss - square_hinge (better) and hinge
	model.fit(trainData, trainLabels)
	print('svm model trained')
	labels = model.predict(testData)
	return labels

'''
best is .769 for penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, 
		fit_intercept=True, intercept_scaling=1, class_weight={0:0.1,1:0.9}, verbose=True, 
		random_state=None, max_iter=
		.7696 penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, 
		fit_intercept=True, intercept_scaling=1, class_weight={0:0.1,1:0.9}, verbose=True, 
		random_state=None, max_iter=2000
for l2 regularization, squared_hinge and dual false, clas weights = (1,10) - 77.83
for l2 regularization, squared_hinge and dual false, clas weights = (1,9) - 77.88
for l2 regularization, squared_hinge and dual false, clas weights = (1,8) - 77.88
for l2 regularization, squared_hinge and dual false, clas weights = (1,7) - 77.63
smote(k=3), same as above, class weights all 1. 76.63
smote(k=2), same as above, class weights all 1. 76.64
smote(k=2), same as above, class weights 1.25,1. 76.64
for undersampling 0.5 and weights = (1,5): accuracy, is 77.59
for undersampling 0.5 and weights = (1,4): accuracy, is 77.58
for undersampling 0.5 and weights = (1,1): accuracy, is 65
for undersampling 0.8 and weights = (1,10): accuracy, is 62
for undersampling 0.8 and weights = (1,1): accuracy, is 75.16
for undersampling 0.2 and weights = (1,10): accuracy, is 77.15
for undersampling 0.2 and weights = (1,8): accuracy, is 77.59
'''

def mlpClassifier(trainData, trainLabels, testData):
	model = MLPClassifier(activation='tanh', alpha=1e-05, batch_size=4000, beta_1=0.9, beta_2=0.999, early_stopping=False,epsilon=1e-08, 
		hidden_layer_sizes=(100, 30), learning_rate='constant', learning_rate_init=0.001, max_iter=200, momentum=0.9, n_iter_no_change=10,
		nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.1, 
		verbose=True, warm_start=False)
	model.fit(trainData, trainLabels)
	print('mlp trained')
	labels = model.predict(testData)
	return labels

from sklearn.ensemble import AdaBoostRegressor

def adaBoostClassifier(trainData, trainLabels, testData):
	model = AdaBoostRegressor(loss = 'linear', n_estimators = 100)
	model.fit(trainData, trainLabels)
	print('ada boost trained')
	labels = model.predict(testData)
	return labels

def classifierCombination(trainData, trainLabels, testData):
	count = 2
	models = [None] * count

	'''
	ratio = i/(count-1.0)
		ratio = 0.10+ratio*(0.20-0.10)
		weights = {0:ratio, 1:(1-ratio)}
	'''
	weights = {0:1,1:10}
	samples = testData.shape[0]
	sums = np.zeros(samples)
	for i in range(count):
		
		if (i==0):
			model = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
				class_weight=weights, verbose=True, random_state=None, max_iter=2000)
			# penalty - l1, l2; loss - square_hinge (better) and hinge
		elif (i==1):
			model = linear_model.LogisticRegression(solver='lbfgs', verbose=True, max_iter=5000, class_weight=weights)
		elif (i==2):
			'''
			model = MLPClassifier(activation='tanh', alpha=1e-05, batch_size=4000, beta_1=0.9, beta_2=0.999, early_stopping=False,epsilon=1e-08, 
				hidden_layer_sizes=100, learning_rate='constant', learning_rate_init=0.001, max_iter=1000, momentum=0.9, 
				nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.1, 
				verbose=True, warm_start=False)
			'''
			labels = neuralNetwork(trainData, trainLabels, testData, 1)
	
		model.fit(trainData, trainLabels)
		models[i] = model
		print('models trained')

		if (i==0):
			labels = models[i].decision_function(testData)
		elif (i==1):
			temp = models[i].predict_proba(testData)
			labels = np.zeros(temp.shape[0])
			for j in range(temp.shape[0]):
				labels[j] = temp[j][1]-temp[j][0]
		#print(labels[0:10])
		sums = [x + y for x, y in zip(sums, labels)]
	
			
	combinedLabels = np.zeros(samples)
	print('--------------------------------------------------')
	for i in range(samples):
		#print('sum - ',sums[i],' ',i),
		if sums[i]>0:
			combinedLabels[i] = 1
		else:
			combinedLabels[i] = 0
	return combinedLabels

	'''
	77.50 for using svm and linearegression classifier combination, no mlp
	'''
from sklearn.model_selection import StratifiedKFold
import time
import lightgbm as lgb
def lgbmclassifier(trainData, trainLabels, testData):
	# read the data into a pandas datafrome
	params = {'objective' : "binary", 
				'boost':"gbdt",
				'metric':"auc",
				'boost_from_average':"false",
				'num_threads':8,
				'learning_rate' : 0.01,
				'num_leaves' : 13,
				'max_depth':-1,
				'tree_learner' : "serial",
				'feature_fraction' : 0.05,
				'bagging_freq' : 5,
				'bagging_fraction' : 0.4,
				'min_data_in_leaf' : 80,
				'min_sum_hessian_in_leaf' : 10.0,
				'verbosity' : 1}
	model_fold = 5
	folds = StratifiedKFold(n_splits=model_fold, shuffle=True, random_state=10)
	y_pred_lgb = np.zeros(len(testData))
	num_round = 1000000
	for fold_n, (train_index, valid_index) in enumerate(folds.split(trainData,trainLabels)):
		print('Fold', fold_n, 'started at', time.ctime())
		X_train, X_valid = trainData[train_index], trainData[valid_index]
		y_train, y_valid = trainLabels[train_index], trainLabels[valid_index]

		train_data = lgb.Dataset(X_train, label=y_train)
		valid_data = lgb.Dataset(X_valid, label=y_valid)


		lgb_model = lgb.train(params,train_data,num_round,#change 20 to 2000
							valid_sets = [train_data, valid_data],verbose_eval=1000,early_stopping_rounds = 3500)##change 10 to 200

		y_pred_lgb += lgb_model.predict(testData, num_iteration=lgb_model.best_iteration)/model_fold

	return y_pred_lgb

	'''
	lgbm = LGBMClassifier(objective = 'binary', metric = 'auc', max_depth = -1, num_leaves = 8, min_data_in_leaf = 25,
		learning_rate = 0.006, bagging_fraction = 0.2, feature_fraction = 0.4, bagging_freq = 1, lambda_l1 = 5,
		lambda_l2 = 5, verbosity = 1, max_bin = 512, num_threads = -1, random_state = 5, boosting_type = 'gbdt')
		gbdt - 54
		goss - 55
		rf - bagging_frac problems
		dart - 50.34
		
		gbdt - varying unmber of leaves
		8 - 50.3
		16 - 52
		32 - 54
		64 - 55.77
		128 - 56.15

		with smote and goss - 60
		real number - 89.5
	'''