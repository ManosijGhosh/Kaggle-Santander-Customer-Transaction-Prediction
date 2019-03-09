import numpy as np
import signal
import sys
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

from confusion import evaluate


def create_model(ip_shape, lrate = 1e-5):

	#initialize data
	#X_test, X_test, Y_test = make_data()

	#initialize parameters
	model = dict()

	dim_input = ip_shape
	layer_1 = 100
	layer_2 = 30
	n_class = 2
	dim_ = dim_input

	#create model
	with tf.name_scope('input'):
		model['ip'] = tf.placeholder(tf.float32, [None, dim_input], name = 'input-vector')
		model['labels'] = tf.placeholder(tf.float32, [None, 2], name='class-labels')

	model['labels'] = tf.stop_gradient(model['labels'])

	# FIRST ENCODING LAYER

	with tf.name_scope('layer-1'):
		model['We_1'] = tf.Variable(tf.random_normal([dim_input, layer_1], stddev=1.0/layer_1), name = 'We-1')
		model['Be_1'] = tf.Variable(tf.random_normal([1, layer_1], stddev=1.0/layer_1), name = 'Be-1')
		model['ye_1'] = tf.nn.tanh(tf.add(tf.matmul(model['ip'], model['We_1']), model['Be_1']), name = 'ye-1')

	
	# SECOND ENCODING LAYER

	with tf.name_scope('layer-2'):
		model['We_2'] = tf.Variable(tf.random_normal([layer_1, layer_2], stddev=1.0/layer_2), name = 'We-2')
		model['Be_2'] = tf.Variable(tf.random_normal([1, layer_2], stddev=1.0/layer_2), name = 'Be-2')
		model['ye_2'] = tf.nn.tanh(tf.add(tf.matmul(model['ye_1'], model['We_2']), model['Be_2']), name = 'ye-2')
	
	# OUPUT LAYER
	#change ye_1 to ye_2 and layer_1 to layer_2, to get 2 hidden layers
	with tf.name_scope('output'):
		model['We_3'] = tf.Variable(tf.random_normal([layer_2, n_class], stddev=1.0/n_class), name = 'We-3')
		model['Be_3'] = tf.Variable(tf.random_normal([1, n_class], stddev=1.0/n_class), name = 'Be-3')
		model['op'] = tf.add(tf.matmul(model['ye_2'], model['We_3']), model['Be_3'])
	
	# LOSS FUNCTION

	with tf.name_scope('loss_optim_4'):

		model['cost'] = tf.nn.softmax_cross_entropy_with_logits_v2(labels=model['labels'], logits=model['op'],name = 'cost')
		model['cost-2'] = tf.nn.softmax(model['op'])
		
		model['optimizer'] = tf.train.AdamOptimizer(lrate).minimize(model['cost'], name = 'optimizer')
		
	# return the model dictionary

	return model;

def train_model(model, trainData, trainLabelsOneHot, batchSize, path_model, path_logdir, epoch = 100):

	with tf.Session() as session:
		init = tf.global_variables_initializer()
		session.run(init)

		#path_model = './model-4'
		#path_logdir = 'logs-auto-4'

		saver = tf.train.Saver()
		writer = tf.summary.FileWriter(path_logdir, session.graph)

		for i in range(epoch):
			loss = np.empty([1,0])
			for count in range(0,trainData.shape[0],batchSize):
				in_vector = trainData[count:min(count+batchSize,trainData.shape[0])]
				label_vector = trainLabelsOneHot[count:min(count+batchSize,trainData.shape[0])]
				#in_vector = in_vector.reshape(1, in_vector.shape[0])
				feed = {model['ip']: in_vector, model['labels']: label_vector}

				_, summary = session.run([model['optimizer'], model['cost']], feed_dict = feed)
				loss = np.append(loss, summary)
				
			#writer.add_summary(summary, i)
			#if (i%10==0):
			print('Epoch: ', i, 'loss - ',np.mean(loss))

		saver.save(session, path_model)

def test_model(model, testData, batchSize, path_model):

	#path_model = './model-4'
	#path_logdir = 'logs-auto-2'

	labels = np.empty([1,0])
	with tf.Session() as sess:

		saver = tf.train.Saver()
		saver.restore(sess, path_model)

		for i in range(0,testData.shape[0],batchSize):

			in_vector = testData[i:min(i+batchSize,testData.shape[0])]
			feed = {model['ip']: in_vector}

			ans = sess.run(model['cost-2'], feed_dict =  feed)

			predict_class = np.argmax(ans, axis=1)
			labels = np.append(labels,predict_class)

	return labels

			
		
def neuralNetwork(trainData, trainLabels, testData, fold):
	
	batchSize = 4000
	lrate = 1e-4
	isSmoteDone = 0
	print('train data size - ',trainData.shape, ' label size - ',trainLabels.shape)

	if (isSmoteDone==1):
		print("Before OverSampling, counts of label '1': {}".format(sum(trainLabels==1)))
		print("Before OverSampling, counts of label '0': {} \n".format(sum(trainLabels==0)))

		sm = SMOTE(random_state = 2, k_neighbors = 3, n_jobs = 4)
		trainData, trainLabels = sm.fit_sample(trainData, trainLabels.ravel())

		print('train data size - ',trainData.shape, ' label size - ',trainLabels.shape)

		print("After OverSampling, counts of label '1': {}".format(sum(trainLabels==1)))
		print("After OverSampling, counts of label '0': {}".format(sum(trainLabels==0)))

	#concersion to one hot encoding
	trainLabelsOneHot = np.zeros([trainLabels.shape[0], 2])
	#print(trainLabels[0:10])
	trainLabelsOneHot[np.arange(trainLabels.shape[0]), trainLabels] = 1
	
	model = create_model(trainData.shape[1], lrate)
	train_model(model, trainData, trainLabelsOneHot, batchSize, path_model='./models_nn/model_kfold_mano_400_%i' %fold, path_logdir='models_nn/logs_kfold_mano_400_%i' %fold, epoch=300)
	labels = test_model(model, testData, batchSize, path_model='./models_nn/model_kfold_mano_400_%i' %fold)
	
	return labels

## THRESH 1.0: true fraud: 434, false fraud: 21397, total: 284800
'''
3 fold, no smote
batch size = 1000
lrate = 1e-4
hidden layers = [120, 80]
epoch = 20
auc = 60.30/59.62

3 fold, no smote
batch size = 1000
lrate = 1e-4
hidden layers = [120, 80]
epoch = 30
auc =  60.55(overfitting)

3 fold, no smote
batch size = 2000
lrate = 1e-4
hidden layers = [120, 80]
epoch = 30
auc =  58.29 (underfitting)

3 fold, smote
batch size = 2000
lrate = 1e-4
hidden layers = [100, 50]
epoch = 200
auc =  61~ (underfitting)

3 fold, no smote:: submitted to kaggle score - 0.626
batch size = 2000
lrate = 1e-4
hidden layers = [100, 30]
epoch = 200
auc =  61.79

3 fold, no smote
batch size = 2000
lrate = 1e-4
hidden layers = [100]
epoch = 200
auc =  61.31

3 fold, no smote
batch size = 4000
lrate = 1e-4
hidden layers = [100, 30]
epoch = 600
auc =  64.31
'''