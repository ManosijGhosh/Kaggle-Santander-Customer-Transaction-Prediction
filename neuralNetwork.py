import numpy as np
import signal
import sys
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from confusion import evaluate


def create_model(ip_shape, batchSize, lrate = 1e-5):

	#initialize data
	#X_test, X_test, Y_test = make_data()

	#initialize parameters
	model = dict()

	dim_input = ip_shape
	layer_1 = 120
	layer_2 = 80
	n_class = 2
	dim_ = dim_input

	#create model
	with tf.name_scope('input'):
		model['ip'] = tf.placeholder(tf.float32, [batchSize, dim_input], name = 'input-vector')
		model['labels'] = tf.placeholder(tf.float32, [batchSize, 1], name='class-labels')

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

	with tf.name_scope('output'):
		model['We_3'] = tf.Variable(tf.random_normal([layer_2, n_class], stddev=1.0/n_class), name = 'We-3')
		model['Be_3'] = tf.Variable(tf.random_normal([1, n_class], stddev=1.0/n_class), name = 'Be-3')
		model['op'] = tf.add(tf.matmul(model['ye_2'], model['We_3']), model['Be_3'])
	
	# LOSS FUNCTION

	with tf.name_scope('loss_optim_4'):

		model['cost'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=model['labels'], logits=model['op']),name = 'cost')
		model['cost-2'] = tf.softmax(model['op'])
		model['sum_loss'] = tf.summary.scalar(model['cost'].name, model['cost'])

		model['optimizer'] = tf.train.AdamOptimizer(lrate).minimize(model['cost'], name = 'optim')
		
	# return the model dictionary

	return model;

def train_model(model, X_train, X_label, batchSize, path_model, path_logdir, epoch = 100):

	with tf.Session() as session:
		init = tf.global_variables_initializer()
		session.run(init)

		#path_model = './model-4'
		#path_logdir = 'logs-auto-4'

		saver = tf.train.Saver()
		writer = tf.summary.FileWriter(path_logdir, session.graph)

		for i in range(epoch):
			for count in range(0,(X_train.shape[0]//batchSize)*batchSize,batchSize):
				in_vector = X_train[count:count+batchSize]
				label_vector = X_label[count:count+batchSize]
				#in_vector = in_vector.reshape(1, in_vector.shape[0])
				feed = {model['ip']: in_vector, model['labels']: label_vector}

				_, summary = session.run([model['optimizer'], model['sum_loss']], feed_dict = feed)
				#print('Cost: ', model['cost'].eval())
			writer.add_summary(summary, i)
			#if (i%10==0):
			print('Epoch: ', i)

		saver.save(session, path_model)

def test_model(model, X_test, batchSize, path_model, outfile):

	#path_model = './model-4'
	#path_logdir = 'logs-auto-2'

	with open(outfile, 'w+') as outfp:
		with tf.Session() as sess:

			saver = tf.train.Saver()
			saver.restore(sess, path_model)

			for i in range(0,(X_test.shape[0]//batchSize)*batchSize,batchSize):

				in_vector = X_test[i:i+batchSize]
				feed = {model['ip']: in_vector}

				outfp.write('Batch --- ' + str(i//batchSize) + ' --- \n')
				ans = sess.run(model['cost-2'], feed_dict =  feed)

				predict_class = np.argmax(ans, axis=1)
				predict_class = predict_class.tolist()

				outfp.write(str(predict_class))
				outfp.write('\n')
		
def neuralNetwork(X_train, X_test, Y_train, str, fold):
	
	#X_test, X_train, Y_test = make_data()
	batchSize = 100
	negatives = list()
	for labels in range(0,X_train.shape[0]):
		if Y_train[labels]==1:
			negatives.append(labels)
	print(len(negatives))
	print('before - ',X_train.shape)
	X_train= np.delete(X_train, negatives, axis=0)
	print('after - ',X_train.shape)

	#'''
	model = create_model(X_train.shape[1], batchSize)
	train_model(model, X_train, batchSize, path_model='./models/model_kfold_mano_400_%i' %fold, path_logdir='models/logs_kfold_mano_400_%i' %fold, epoch=50)
	test_model(model, X_test, batchSize, path_model='./models/model_kfold_mano_400_%i' %fold, outfile=('models/'+str+'_%i') %fold)
	#'''

## THRESH 1.0: true fraud: 434, false fraud: 21397, total: 284800