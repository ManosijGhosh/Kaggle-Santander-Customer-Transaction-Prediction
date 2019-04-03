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
	n_class = 1
	dim_ = dim_input

	#create model
	with tf.name_scope('input'):
		model['ip'] = tf.placeholder(tf.float32, [None, dim_input], name = 'input-vector')
		model['labels'] = tf.placeholder(tf.float32, [None,n_class], name='class-labels')

	model['labels'] = tf.stop_gradient(model['labels'])

	# FIRST ENCODING LAYER

	with tf.name_scope('layer-1'):
		model['We_1'] = tf.Variable(tf.random_normal([dim_input, layer_1], stddev = 0.2), name = 'We-1')
		model['Be_1'] = tf.Variable(tf.random_normal([1, layer_1], stddev = 0.2), name = 'Be-1')
		model['ye_1'] = tf.nn.relu(tf.add(tf.matmul(model['ip'], model['We_1']), model['Be_1']), name = 'ye-1')

	
	# SECOND ENCODING LAYER

	with tf.name_scope('layer-2'):
		model['We_2'] = tf.Variable(tf.random_normal([layer_1, layer_2], stddev = 0.2), name = 'We-2') #stddev=1.0/layer_2)
		model['Be_2'] = tf.Variable(tf.random_normal([1, layer_2], stddev = 0.2), name = 'Be-2')
		model['ye_2'] = tf.nn.relu(tf.add(tf.matmul(model['ye_1'], model['We_2']), model['Be_2']), name = 'ye-2')
	
	# OUPUT LAYER
	#change ye_1 to ye_2 and layer_1 to layer_2, to get 2 hidden layers
	with tf.name_scope('output'):
		model['We_3'] = tf.Variable(tf.random_normal([layer_2, n_class], stddev = 0.2), name = 'We-3')
		model['Be_3'] = tf.Variable(tf.random_normal([1, n_class], stddev = 0.2), name = 'Be-3')
		model['op'] = tf.nn.sigmoid(tf.add(tf.matmul(model['ye_2'], model['We_3']), model['Be_3']))
	
	print(model['op'].shape)
	# LOSS FUNCTION

	with tf.name_scope('loss_optim_4'):

		# this is the weight for each datapoint, depending on its label

		#model['cross-entropy'] =  tf.nn.softmax_cross_entropy_with_logits_v2(labels=model['labels'], logits=model['op'], name='cross-entropy') #shape [batch_size,1]
		#model['cross-entropy'] =  tf.nn.softmax_cross_entropy_with_logits_v2model(['cross-entropy'] = , name='cross-entropy') #shape [batch_size,1]
		model['cross-entropy'] = tf.losses.absolute_difference(labels=model['labels'], predictions=model['op'])
		model['cost'] = tf.reduce_mean(model['cross-entropy'],name = 'cost')

		print(model['labels'].shape, ' - ', model['op'].shape)
	
		model['cost-2'] = tf.metrics.auc(labels=model['labels'], predictions = model['op'], num_thresholds = 200, name='cost-2') #shape [1, batch_size]
		model['ce'] = tf.losses.absolute_difference(labels=model['labels'], predictions=model['op'])
		model['optimizer'] = tf.train.AdamOptimizer(lrate).minimize(model['cost'], name = 'optimizer')
		
	# return the model dictionary

	return model

def train_model(model, trainData, trainLabelsOneHot, batchSize, path_model, path_logdir, epoch = 100):

	with tf.Session() as session:
		init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		session.run(init)

		#path_model = './model-4'
		#path_logdir = 'logs-auto-4'

		saver = tf.train.Saver()
		writer = tf.summary.FileWriter(path_logdir, session.graph)

		for i in range(epoch):
			loss = np.empty([1,0])
			loss_ce = np.empty([1,0])
			for count in range(0,trainData.shape[0],batchSize):
				in_vector = trainData[count:min(count+batchSize,trainData.shape[0])]
				label_vector = trainLabelsOneHot[count:min(count+batchSize,trainData.shape[0])]
				#print(label_vector.shape)
				label_vector = np.reshape(label_vector, (label_vector.shape[0],1))
				#print(label_vector.shape)
				#in_vector = in_vector.reshape(1, in_vector.shape[0])
				feed = {model['ip']: in_vector, model['labels']: label_vector}

				_, lauc, lce = session.run([model['optimizer'], model['cost-2'], model['ce']], feed_dict = feed)
				loss = np.append(loss, lauc)
				loss_ce = np.append(loss_ce, lce)
				
			#writer.add_summary(summary, i)
			#if (i%10==0):
			print('Epoch: ', i, 'ce loss - ', np.mean(loss_ce), ' auc loss - ',np.mean(loss))

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

			ans = sess.run(model['op'], feed_dict =  feed)

			labels = np.append(labels,ans)

	return labels

			
		
def neuralNetworkAuc(trainData, trainLabels, testData, fold):
	
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

	model = create_model(trainData.shape[1], lrate)
	train_model(model, trainData, trainLabels, batchSize, path_model='./models_nn/model_kfold_mano_400_%i' %fold, path_logdir='models_nn/logs_kfold_mano_400_%i' %fold, epoch=50)
	labels = test_model(model, testData, batchSize, path_model='./models_nn/model_kfold_mano_400_%i' %fold)
	
	return labels

