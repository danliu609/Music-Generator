#!/User/liudan/tensorflow//bin/python
# Filename: RNN.py
import math
import re
import tensorflow as tf
import numpy as np
import collections
import cPickle
import random
#this is train the rnn networks to predict the melody from the melody dataset

#setup size parameters
#vocabulary = list(set(train_word))
#vocaSz = 8000#len(vocabulary)
#trainSz = len(train_word)
#testSz = len(test_word)
embed_size = 50
batchSz = 10
stepNum = 64
LSTM_cellSz = 256
vecDim = 35

#get batch size of data
def get_batch(index, batch_size, num_steps, data_set):
    batchrow = data_set[index:index+batch_size*num_steps]
    #batch = np.reshape(batchrow, [batch_size, num_steps, vecDim])
    batchrow = np.reshape(batchrow, [batch_size, num_steps])
    return batchrow
    
#dataset: melody

def getbatch_random(batchsz, num_steps, dataset):
    batch = []
    batch_y = []
    i = 0
    while i < batchsz:
        #print "i", i
        index = random.randint(0, len(dataset)-1)
        #print "index ", index
        #print "length", len(dataset[index])
        if len(dataset[index]) >= num_steps:
            matrix = dataset[index]
            start = random.randint(0, len(matrix)- num_steps -1)
            #print "start ", start
            batch.append(matrix[start:start+num_steps])
            batch_y.append(matrix[start+1:start+num_steps+1])
            i = i+1
    output = np.reshape(batch,[batchsz, num_steps])
    output_y = np.reshape(batch_y, [batchsz, num_steps])
    return output, output_y
    

#read input file    
with open('melody_index.pckl','r') as f:
    #inputs = cPickle.load(f)
    input_data = cPickle.load(f)



print 'inputs size is ', len(input_data)


sess = tf.InteractiveSession()

#setup model placeholders
INPT = tf.placeholder(tf.int32,[batchSz, stepNum])
OUTPT = tf.placeholder(tf.int32,[batchSz, stepNum])
keep_prob = tf.placeholder(tf.float32)

#initialize model parameters
E = tf.Variable(tf.truncated_normal([vecDim, embed_size], stddev=0.1))
#E = tf.Variable(tf.random_uniform([vocaSz, embedSz], -1.0, 1.0))

BLTM = tf.nn.rnn_cell.BasicLSTMCell(LSTM_cellSz)
InitState = BLTM.zero_state(batchSz, tf.float32)
weights = tf.ones([batchSz*stepNum])

W = tf.Variable(tf.truncated_normal([LSTM_cellSz, vecDim],stddev=0.1))
b = tf.Variable(tf.truncated_normal([vecDim],stddev=0.1))

#forward pass
#get the embedding from the E matrix
embd = tf.nn.embedding_lookup(E, INPT)
#drop out some embedding
embed = tf.nn.dropout(embd, keep_prob)

rnn_output, outstate = dyrnn =tf.nn.dynamic_rnn(BLTM, embed,initial_state = InitState)
h = tf.reshape(rnn_output,[batchSz*stepNum,LSTM_cellSz])
logits = tf.matmul(h,W)+b

#setup loss function
Ys = tf.reshape(OUTPT, shape = [batchSz*stepNum])
cross_entropy = tf.reduce_mean(tf.nn.seq2seq.sequence_loss_by_example([logits], [Ys], [weights]))

#Setup the training operation
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

saver = tf.train.Saver(tf.all_variables())
#training
print 'Start training ...'
sess.run(tf.initialize_all_variables())
NUM_EPOCHS = 5

state_new = sess.run(InitState)
for e in range(NUM_EPOCHS):
    num = 0
    error_Total = 0
    print 'num e training', e
    #while x < trainSz//(stepNum*batchSz)*(stepNum*batchSz)-1:
    for i in range(len(input_data)):
        x = 0
        
        while x<10:
            
            train_in_batch, train_out_batch = getbatch_random(batchSz, stepNum, input_data)        
            x = x+1
            num = num+1
            dyrnnNew,_,error = sess.run([dyrnn,train_step,cross_entropy],feed_dict={INPT:train_in_batch,OUTPT: train_out_batch,keep_prob:0.5, InitState: state_new})
            state_new = dyrnnNew[1]
            error_Total += error 
        #if x%100 == 0:
        print 'midi',i,' error is ',error
    print "perplexity", math.exp(error_Total/num)
    saver.save(sess, 'music-model-random', global_step=e) 
  
print 'Training Done!'
