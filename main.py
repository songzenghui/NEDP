import tensorflow as tf 
import numpy as np 
import random
import networkx as nx 
import scipy.io as sio 
import os,sys
import pickle
import walk

def get_batch(arr,n_seqs,n_steps):
#	arr = arr.reshape([-1])
#	batches = n_steps * n_seqs
#	n_batches = int(len(arr) / batches)

#	arr = arr[:n_batches*batches]
#	arr = arr.reshape([n_seqs,-1])
	n = int(arr.shape[0]/n_seqs) * n_seqs
	nn = int(arr.shape[1]/n_steps) * n_steps
	arr = arr[:n,:nn]

	for n in range(0,arr.shape[0],n_seqs):
		for nn in range(0,arr.shape[1],n_steps):
			
	#		x = arr[:,n:n+n_steps]
			x = arr[n:n+n_seqs,nn:nn+n_steps]
			y = np.zeros_like(x)
			y[:,:-1],y[:,-1] = x[:,1:],x[:,0]
			yield x,y

def build_encode_arr(corpus,vocab_to_int):
	encode_input = np.zeros([len(corpus),len(corpus[0])],dtype=np.int32)
	for i,path in enumerate(corpus):
		tmp = list(map(lambda x:vocab_to_int[x],path))
		encode_input[i] = np.array(tmp,dtype=np.int32)
	# encode_output = np.transpose(encode_input)
	# np.random.shuffle(encode_output)

	# return encode_input,np.transpose(encode_output)
	return encode_input

def input_layer(n_steps,n_seqs):

	input = tf.placeholder(dtype=tf.int32,shape=(n_seqs,n_steps),name='input')
	targets = tf.placeholder(dtype=tf.int32,shape=(n_seqs,n_steps),name='targets')
	keep_prob = tf.placeholder(dtype=tf.float32,name='keep_prob')

	return input,targets,keep_prob


def basic_cell(lstm_cell_num,keep_prob,mode,n_seqs,n_steps,d):
	if mode == 'lstm':
		lstm = tf.contrib.rnn.LSTMCell(lstm_cell_num,initializer=tf.orthogonal_initializer,forget_bias=0.0)
	elif mode == 'rnn':
		lstm = tf.contrib.rnn.BasicRNNCell(lstm_cell_num)
	else:
		raise Exception('Unkown mode:{},only support rnn and lstm mode'.format(mode))

	return tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob=keep_prob)

def hidden_layer(lstm_cell_num,lstm_layer_num,n_seqs,keep_prob,mode,n_steps,d):

	multi_lstm = tf.contrib.rnn.MultiRNNCell([basic_cell(lstm_cell_num,keep_prob,mode,n_seqs,n_steps,d) for _ in range(lstm_layer_num)])
	initial_state = multi_lstm.zero_state(n_seqs,tf.float32)

	return multi_lstm,initial_state


def output_layer(lstm_output,in_size,out_size):

	out = tf.concat(lstm_output,1)
	x = tf.reshape(out,[-1,in_size])

	with tf.variable_scope('softmax'):
		W = tf.Variable(tf.truncated_normal([in_size,out_size],stddev=0.1))
		b = tf.Variable(tf.zeros(out_size))

	logits = tf.matmul(x,W) + b
	prob_distrib = tf.nn.softmax(logits,name='predictions')

	return logits,prob_distrib

def cal_loss(logits,targets,class_num,t,embedding,alpha):

	y_one_hot = tf.one_hot(targets,class_num)
	y = tf.reshape(y_one_hot,logits.get_shape())
	# y = tf.reshape(targets,[logits.get_shape()[0],-1])

	# loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits/t,labels=y) + alpha*tf.nn.l2_loss(embedding)
	loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=y)

	return tf.reduce_mean(loss)

def optimizer(loss,learning_rate,grad_clip):
	with tf.name_scope('gradient'):

		tvars = tf.trainable_variables()  #return the trainable variables,not all variables
		unclip_grad = tf.gradients(loss,tvars)
		grad,_ = tf.clip_by_global_norm(unclip_grad,grad_clip)
		# tf.summary.histogram('unclip_grad',unclip_grad)
		# tf.summary.histogram('grad',grad)

	train_op = tf.train.AdamOptimizer(learning_rate)

	return train_op.apply_gradients(zip(grad,tvars))

class LSTM:
	def __init__(self,class_num,n_steps,n_seqs,lstm_cell_num,lstm_layer_num,learning_rate,grad_clip,mode,d,t,alpha):
		tf.reset_default_graph() 
		#input layer
		self.input,self.targets,self.keep_prob = input_layer(n_steps,n_seqs)

		#lstm layer
		lstm_cell , self.initial_state = hidden_layer(lstm_cell_num,lstm_layer_num,n_seqs,self.keep_prob,mode,n_steps,d)

		with tf.variable_scope('embedding_layer'):
				embedding = tf.get_variable(name='embedding',shape=[class_num,d],initializer=tf.random_uniform_initializer())

		x_input = tf.nn.embedding_lookup(embedding,self.input)
		if mode == 'clstm':
			x_input = tf.expand_dims(x_input,-1)
			# x_input_ = tf.unstack(x_input[...,None],128,2)

		self.embedding = embedding

		output,state = tf.nn.dynamic_rnn(lstm_cell,x_input,initial_state=self.initial_state)
		# output,state = tf.nn.dynamic_rnn(lstm_cell,x_input_,initial_state=self.initial_state)
		self.out = output
		self.final_state = state

		self.logits,self.pred  = output_layer(output,lstm_cell_num,class_num)

		with tf.name_scope('loss'):
			self.loss = cal_loss(self.logits,self.targets,class_num,t,x_input,alpha)

		self.optimizer = optimizer(self.loss,learning_rate,grad_clip)

class Dis:
	def __init__(self,lr,class_num,d,beta):
		self.adj = tf.placeholder(dtype=tf.float32,name='adj')
		self.index = tf.placeholder(dtype=tf.int32,name='index')

		with tf.name_scope('rep'):
			with tf.variable_scope('embedding_layer',reuse=True):
				embedding = tf.get_variable(name='embedding',shape=[class_num,d],initializer=tf.random_uniform_initializer)
			# tf.summary.histogram('rep',embedding)

		D = tf.diag(tf.reduce_sum(self.adj,1))
		L = D - self.adj

		batch_emb = tf.nn.embedding_lookup(embedding,self.index)

		with tf.name_scope('lap_loss'):
			self.lap_loss = 2*tf.trace(tf.matmul(tf.matmul(tf.transpose(batch_emb),L),batch_emb)) + beta*tf.nn.l2_loss(batch_emb)
		tvars = tf.trainable_variables('embedding_layer')
		grad = tf.gradients(self.lap_loss,tvars)
		
		self.lap_optimizer = tf.train.RMSPropOptimizer(lr).apply_gradients(zip(grad,tvars))

if __name__ == '__main__':

	tf.app.flags.DEFINE_string('format','mat','the file format')
	tf.app.flags.DEFINE_string('input','blogcatalog.mat','the file name')
	tf.app.flags.DEFINE_string('output','rep.npy','saved as npy')
	tf.app.flags.DEFINE_string('mode','lstm','which mode will used')
	tf.app.flags.DEFINE_integer('node_num',100,'nums of per node')
	tf.app.flags.DEFINE_integer('path_length',100,'length of path')
	tf.app.flags.DEFINE_integer('timesteps',100,'length of one sequence')
	tf.app.flags.DEFINE_integer('sequences',100,'how many sequences in one batch')
	tf.app.flags.DEFINE_integer('hidden_size',512,'hidden size')
	tf.app.flags.DEFINE_integer('batches',128,'batches size')
	tf.app.flags.DEFINE_integer('layer',1,'how many layers')
	tf.app.flags.DEFINE_integer('representation_size',128,'representation size')
	tf.app.flags.DEFINE_float('t',1.0,'smooth softmax')
	tf.app.flags.DEFINE_float('alpha',1.0,'LSTM L2 reg ')
	tf.app.flags.DEFINE_float('beta',1.0,'Lap L2 reg')
	tf.app.flags.DEFINE_float('lr',0.001,'learning rate')
	tf.app.flags.DEFINE_float('lap_lr',0.001,'learning rate')
	tf.app.flags.DEFINE_float('keep_prob',0.5,'keep prob')
	tf.app.flags.DEFINE_integer('grad_clip',5,'gradient clipping')
	tf.app.flags.DEFINE_integer('gen_epoches',5,'iter nums')
	tf.app.flags.DEFINE_integer('dis_epoches',5,'iter nums')
	tf.app.flags.DEFINE_integer('epoches',5,'iter nums')
	FLAGS = tf.app.flags.FLAGS

	if FLAGS.format == 'mat':
		mat = sio.loadmat(FLAGS.input)['network']
		G = nx.from_scipy_sparse_matrix(mat)
	elif FLAGS.format == 'adjlist':
		G = nx.read_adjlist(FLAGS.input)
	elif FLAGS.format == 'edgelist':
		G = nx.read_edgelist(FLAGS.input)
	else:
		raise Exception("Unkown file format:{}.Valid format is 'mat','adjlist','edgelist'".format(FLAGS.format) )
	mat = nx.to_scipy_sparse_matrix(G)	



	print('Start Walking...\n **********')
	for edge in G.edges():
		G[edge[0]][edge[1]]['weight'] *= min(G.degree(edge[0]),G.degree(edge[1])) / max(G.degree(edge[0]),G.degree(edge[1]))
	G_ = walk.Walk(G,False,0.25,0.25)
	G_.preprocess_transition_probs()
	corpus = G_.simulate_walks(FLAGS.node_num, FLAGS.path_length)
	vocab = list(G.nodes())
	vocab_to_int = {c:i for i,c in enumerate(vocab)}
	int_to_vocab = dict(enumerate(vocab))

	encode_arr_input = build_encode_arr(corpus,vocab_to_int)

	lstm = LSTM(len(vocab),FLAGS.timesteps,FLAGS.sequences,FLAGS.hidden_size,FLAGS.layer,FLAGS.lr,FLAGS.grad_clip,FLAGS.mode,FLAGS.representation_size,FLAGS.t,FLAGS.alpha)
	dis = Dis(FLAGS.lap_lr,len(vocab),FLAGS.representation_size,FLAGS.beta)
	# saver = tf.train.Saver()

	config = tf.ConfigProto()  
	config.gpu_options.allow_growth=True  
	sess = tf.Session(config=config)  

	sess.run(tf.global_variables_initializer())
	print('Launch LSTM...')

	for i in range(FLAGS.epoches):

		for j in range(FLAGS.gen_epoches):
			
			new_state = sess.run(lstm.initial_state,{lstm.keep_prob:FLAGS.keep_prob})

			for x,y in get_batch(encode_arr_input,FLAGS.sequences,FLAGS.timesteps):
				feed_dict = {lstm.input:x,lstm.targets:y,lstm.keep_prob:FLAGS.keep_prob,lstm.initial_state:new_state}
				batch_loss,new_state,_ = sess.run([lstm.loss,lstm.final_state,lstm.optimizer],feed_dict=feed_dict)

		for k in range(FLAGS.dis_epoches):
			adj = mat.toarray()
			for index in range(0,adj.shape[0],FLAGS.batches):
				batch_adj = adj[index:index+FLAGS.batches,index:index+FLAGS.batches]

				feed_dict = {dis.adj:batch_adj,dis.index:np.arange(adj.shape[0])[index:index+FLAGS.batches]}
				lap_loss,_ = sess.run([dis.lap_loss,dis.lap_optimizer],feed_dict=feed_dict)

	np.save(FLAGS.output,sess.run(lstm.embedding))


	
