import tensorflow as tf

import numpy as np
from scipy.misc import imresize

import matplotlib as m
m.use('Agg')
import matplotlib.pyplot as plt
from skimage import io
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tqdm import tqdm

seed = 256

directory = os.listdir('data/')

fonts = [os.path.join('data', d+'/') for d in os.listdir('data/')]
data_links = sorted([os.path.join(f, i) for f in fonts for i in os.listdir(f)])

labels = np.array([np.zeros(1300, dtype=np.int32)+i for i in range(0, 10)]).flatten()

data_raw = np.array([io.imread(img, as_grey=True).astype(np.float32) for img in data_links])

data = np.array([imresize(img, (28, 126)) for img in data_raw])
data = np.array([img[:,:-26] for img in data])

def onehot(y):
    a = np.zeros((y.shape[0], 10))
    a[np.arange(y.shape[0]), y] = 1
    print(a[0])
    return a

y = onehot(labels)
train = data[:,:,:,np.newaxis] / 255
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.25, random_state=seed)
print(X_train.shape)
print(X_test.shape)
print(np.sum(y_train, axis=0))

def conv_relu(inputs, filters, k_size, stride, padding, scope_name):
    '''
    A method that does convolution + relu on inputs
    '''
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        num_channels = inputs.shape[-1]
        kernel = tf.get_variable(dtype=tf.float32, 
                                  shape=[k_size, k_size, num_channels, filters], 
                                  initializer = tf.random_normal_initializer(),
                                  name='kernel_'+ scope.name)
        
        biases = tf.get_variable(dtype=tf.float32, 
                                 shape=[filters],
                                 initializer=tf.random_normal_initializer(),
                                 name='biases_' + scope.name)
        
        conv = tf.nn.conv2d(inputs, 
                            kernel, 
                            strides=[1, stride, stride, 1], 
                            padding=padding)
        
        activation = tf.nn.relu(conv + biases, name=scope.name)
         
    return activation

#--------------------------------------------------------------------------------

def maxpool(inputs, ksize, stride, padding='VALID', scope_name='pool'):
    '''A method that does max pooling on inputs'''
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool = tf.nn.max_pool(inputs, 
                              ksize=[1, ksize, ksize, 1], 
                              strides=[1,stride,stride,1],
                              padding=padding,
                              name='max_'+scope.name)
    
    return pool

#--------------------------------------------------------------------------------

def fully_connected(inputs, out_dim, scope_name='fc'):
    '''
    A fully connected linear layer on inputs
    '''
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        #print(inputs)
        in_dim = inputs.shape[-1]
        weights = tf.get_variable(dtype=tf.float32, 
                                  initializer=tf.random_normal_initializer(),
                                  shape=[in_dim, out_dim],
                                  name='weights_'+scope.name)
        biases = tf.get_variable(dtype=tf.float32, 
                                 shape=[out_dim],
                                 initializer=tf.constant_initializer(0.0),
                                 name='biases_'+ scope.name)
    
        out = tf.matmul(inputs,weights) + biases
        
    return out


learning_rate = 0.01
batch_size = 128
n_epochs = 12
n_classes = 10
global_step = tf.train.get_or_create_global_step()  

X = tf.placeholder(name='Input', dtype='float32', shape=[None, 28, 100, 1])
y = tf.placeholder(name='Target', dtype='float32', shape = [None, n_classes])



conv_1 = conv_relu(scope_name='conv_1', 
                   inputs=X, 
                   filters=32, 
                   stride=1,
                   k_size=5, 
                   padding='SAME')


conv_2 = conv_relu(scope_name='conv_2', 
                   inputs=conv_1, 
                   filters=64, 
                   stride=1,
                   k_size=5, 
                   padding='SAME')


pool_1 = maxpool(scope_name='maxpool_1', 
                 inputs=conv_2, 
                 ksize=2, 
                 stride=2, 
                 padding='VALID')



flatten = pool_1.shape[1] * pool_1.shape[2] * pool_1.shape[3]
pool_1 = tf.reshape(pool_1, [-1, flatten])


fc_1 = fully_connected(pool_1, out_dim=2048, scope_name='fully_connected_1')


dropout_2 = tf.nn.dropout(x=tf.nn.relu(fc_1), keep_prob=0.5, name='dropout_2')


logits = fully_connected(inputs=dropout_2, out_dim=n_classes, scope_name='logits_fc')



with tf.name_scope('loss'):
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels= y, logits= logits)
    loss = tf.cast(tf.reduce_mean(entropy), tf.float32)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))    
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)


    
      
results = np.zeros(n_epochs)
    
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        
        idx = np.random.permutation(range(len(X_train)))
        n_batches = len(X_train) // batch_size
        
        for batch in tqdm(range(n_batches)):
            idx_i = idx[batch * batch_size : (batch+1) * batch_size]
            sess.run(optimizer, feed_dict={X: X_train[idx_i], y: y_train[idx_i]})
            #print(batch,'/',n_batches)
        loss_i, acc = sess.run([loss, accuracy], feed_dict={X: X_train, y: y_train})
        results[epoch] = loss_i
        
        print('Epoch:', epoch)
        print('Loss:', loss_i)
        print('Train accuracy:', acc)
        print('-'*30)
        
    print('Final score:', accuracy_score(y_test, sess.run(logits, {X: X_test})))










