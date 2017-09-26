import numpy as np 
import tensorflow as tf


def read_csv(filename,sess):

	filename_queue = tf.train.string_input_producer([filename],shuffle=False)

	reader=tf.TextLineReader(skip_header_lines=1)	
	key,value = reader.read(filename_queue)

	record_defaults = [[0.0],[0.0],[0.0],[0.0],[0]]

	col1,col2,col3,col4,col5 = tf.decode_csv(value,record_defaults=record_defaults)
	features = tf.stack([col1,col2,col3,col4])
	
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess)
	
	return tf.cast(features,dtype=tf.float32),tf.cast(col5,dtype=tf.float32)



epochs = 50
lr = 0.01


def update_w(x,t,w):
	a_f = tf.multiply(lr,tf.scalar_mul(t,x))
	r = tf.add(w,a_f)
	return r;

def update_b(b,t,w):
	r = tf.add(b,tf.multiply(lr,t))
	return r

def perceptron():

	w = tf.Variable(tf.random_normal([4]))
	b = tf.Variable(0.0)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		x,t=read_csv('iris_dataset.csv',sess)

		for epoch in range(epochs):
			for i in range(150):
				
				z = tf.add(tf.reduce_sum(tf.multiply(w,x)),b)
				y = tf.cond(tf.less(z,tf.constant(0.0)),lambda:tf.constant(-1.0),lambda:tf.constant(1.0))

				w = tf.cond(tf.equal(y,t),lambda:w,lambda:update_w(x,t,w))
				b = tf.cond(tf.equal(y,t),lambda:b,lambda:update_b(x,t,b))

				w1,z1 = sess.run([w,z])
				print w1 , z1

			print ("Completed epoch : ",epoch," of ",epochs)

perceptron()

