from sklearn.neighbors import KNeighborsClassifier

def createKnnModel(trainData, trainLabels, testData):
	knnModel = KNeighborsClassifier(n_neighbors=3, n_jobs = 4)
	knnModel.fit(trainData, trainLabels)
	print('knnTrained')
	results = knnModel.predict(testData)
	return results