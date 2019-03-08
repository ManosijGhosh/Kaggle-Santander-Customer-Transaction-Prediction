import numpy as np

from sklearn import linear_model, datasets, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

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
	logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial')
	
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
	randomForest = RandomForestClassifier(n_jobs = 4)
	randomForest.fit(trainData, trainLabels)
	print('rabdom forest trained')
	labels = randomForest.predict(testData)
	return labels

def featureSelectionMI(trainData, trainLabels, testData, testLabels):
	n = trainData.shape[0]
	miValue=np.repeat(0.0,n)
	for i in range(0,n-1):
		miValue[i] = metrics.normalized_mutual_info_score(trainData[:,i],trainLabels)
		
	print(miValue)
	ranking=(-miValue).argsort()
	print(ranking)

	for loopval in range(10,200,10):
		temp=ranking[0:loopval]
		#print(temp)
		traindatanew=trainData[:,temp]
		testdatanew=testData[:,temp]
		labels = logisticRegression(datanew, trainLabels, testdatanew)
		auc = consfusion_eval(testLabels, rfLabels)
		print('for ',loopval,' features; AUC - ',auc)

