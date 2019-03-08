#NOT WORKING
import numpy as np
import numpy as np

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import clone

'''
def restrictedBoltzmannMachine(trainData, trainLabels, testData):
	model = BernoulliRBM(n_components = 100, learning_rate = 1e-2,batch_size = 2000, n_iter = 2, verbose = 1)
	model.fit(trainData, trainLabels)
	print('boltzmann trained')
	labels = model.score_samples(testData)
	print(labels[1:50])

	labels = np.argmax(labels, axis=1)
	return labels
'''
def restrictedBoltzmannMachine(trainData, trainLabels, testData):
	logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=10000, multi_class='multinomial')
	rbm = BernoulliRBM(random_state=0, batch_size = 2000, verbose=True)

	rbm_features_classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

	# #############################################################################
	# Training

	# Hyper-parameters. These were set by cross-validation,
	# using a GridSearchCV. Here we are not performing cross-validation to
	# save time.
	rbm.learning_rate = 0.06
	rbm.n_iter = 20
	# More components tend to give better prediction performance, but larger
	# fitting time
	rbm.n_components = 100
	logistic.C = 6000

	# Training RBM-Logistic Pipeline
	rbm_features_classifier.fit(trainData, trainLabels)
	labels = rbm_features_classifier.predict(testData)

	#labels = list(labels)
	return labels

	'''
	# Training the Logistic regression classifier directly on the pixel
	raw_pixel_classifier = clone(logistic)
	raw_pixel_classifier.C = 100.
	raw_pixel_classifier.fit(trainData, trainLabels)

	# #############################################################################
	# Evaluation

	labels = rbm_features_classifier.predict(testData)
	print("Logistic regression using RBM features:\n%s\n" % (metrics.classification_report(Y_test, Y_pred)))

	Y_pred = raw_pixel_classifier.predict(X_test)
	print("Logistic regression using raw pixel features:\n%s\n" % (
	    metrics.classification_report(Y_test, Y_pred)))
	'''