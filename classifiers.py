import numpy as np

from sklearn import linear_model, datasets, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import pearsonr
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier


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


def logisticRegression(trainData, trainLabels, testData):
	logistic = linear_model.LogisticRegression(solver='lbfgs', verbose=True, max_iter=5000, class_weight={0:1,1:10})
	
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
	randomForest = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, 
		fit_intercept=True, intercept_scaling=1, class_weight={0:0.11,1:0.89}, verbose=True, 
		random_state=None, max_iter=2000)
	# penalty - l1, l2; loss - square_hinge (better) and hinge
	randomForest.fit(trainData, trainLabels)
	print('rabdom forest trained')
	labels = randomForest.predict(testData)
	return labels

'''
best is .769 for penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, 
		fit_intercept=True, intercept_scaling=1, class_weight={0:0.1,1:0.9}, verbose=True, 
		random_state=None, max_iter=
		.7696 penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, 
		fit_intercept=True, intercept_scaling=1, class_weight={0:0.1,1:0.9}, verbose=True, 
		random_state=None, max_iter=2000


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
	count = 3
	models = [None] * count

	'''
	ratio = i/(count-1.0)
		ratio = 0.10+ratio*(0.20-0.10)
		weights = {0:ratio, 1:(1-ratio)}
	'''
	weights = {0:1,1:10}
	for i in range(count):
		
		if (i==0):
			model = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
				class_weight=weights, verbose=True, random_state=None, max_iter=2000)
			# penalty - l1, l2; loss - square_hinge (better) and hinge
		elif (i==1):
			model = linear_model.LogisticRegression(solver='lbfgs', verbose=True, max_iter=5000, class_weight=weights)
		elif (i==2):
			model = MLPClassifier(activation='tanh', alpha=1e-05, batch_size=4000, beta_1=0.9, beta_2=0.999, early_stopping=False,epsilon=1e-08, 
				hidden_layer_sizes=100, learning_rate='constant', learning_rate_init=0.001, max_iter=200, momentum=0.9, n_iter_no_change=10,
				nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.1, 
				verbose=True, warm_start=False)
	
		model.fit(trainData, trainLabels)
		models[i] = model
	print('all svm models trained')
	samples = testData.shape[0]
	sums = np.zeros(samples)
	print(sums)
	for i in range(count):
		if (i==0):
			labels = models[i].decision_function(testData)
		elif (i==2 or i==1):
			temp = models[i].predict_proba(testData)
			labels = np.zeros(temp.shape[0])
			for j in range(temp.shape[0]):
				labels[j] = temp[j][1]-temp[j][0]
		#print(labels[0:10])
		sums = [x + y for x, y in zip(sums, labels)]
	combinedLabels = np.zeros(samples)
	print('---------------------------------------')
	for i in range(samples):
		#print('sum - ',sums[i],' ',i),
		if sums[i]>0:
			combinedLabels[i] = 1
		else:
			combinedLabels[i] = 0
	return combinedLabels