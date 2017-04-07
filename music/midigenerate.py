#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:27:02 2016

@author: ztan6
"""

#!/User/liudan/tensorflow//bin/python
# Filename: RNN.py
import math
import re
import tensorflow as tf
import numpy as np
import collections
import cPickle
import midi
import random

#this python file is used to generate melody from melody using rnn networks
#melody firstly  index
#then usd embedding matrix just like hw5
#when generate, random a notes to start, and then get the result to be the next input

vecDim = 35

#setup size parameters
#vocabulary = list(set(train_word))
#vocaSz = 8000#len(vocabulary)
#trainSz = len(train_word)
#testSz = len(test_word)
upperBound = 89
lowerBound = 55
span = upperBound-lowerBound
#Matrix to midi
def noteStateMatrixToMidi(statematrix, name, span=span):
    statematrix = np.array(statematrix)
    if not len(statematrix.shape) == 3:
        statematrix = np.dstack((statematrix[:, :span], statematrix[:, span:]))
    statematrix = np.asarray(statematrix)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)
    
    span = upperBound-lowerBound
    tickscale = 55
    
    lastcmdtime = 0
    prevstate = [[0,0] for x in range(span)]
    for time, state in enumerate(statematrix + [prevstate[:]]):  
        offNotes = []
        onNotes = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, pitch=note+lowerBound))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=40, pitch=note+lowerBound))
            lastcmdtime = time
            
        prevstate = state
    
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
    midi.write_midifile("{}.mid".format(name), pattern)
#get batch size of data
def get_batch(index, batch_size, num_steps, data_set):
    batchrow = data_set[index:index+batch_size*num_steps]
    batch = np.reshape(batchrow, [batch_size, num_steps, vecDim])
    return batch

embed_size = 50
vecDim = 35
batchSz = 1
#stepNum = len(inputs[0])
step_Num = 1
LSTM_cellSz = 256
#vecDim = 176

sess = tf.InteractiveSession()

#setup model placeholders
INPT = tf.placeholder(tf.int32,[batchSz, None])
OUTPT = tf.placeholder(tf.int32,[batchSz, None])
keep_prob = tf.placeholder(tf.float32)
stepNum = tf.placeholder(tf.int32)
#initialize model parameters
E = tf.Variable(tf.truncated_normal([vecDim, embed_size], stddev=0.1))


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
saver.restore(sess, 'music-model-random-4')

generate_get = []
for i in range(step_Num*batchSz):
    generate_get.append(random.randint(0, span-1))

songMatrix = []
for i in generate_get:
    songMatrix.append(i)
generate_get = np.reshape(generate_get, [batchSz, step_Num])
print songMatrix
state_new = sess.run(InitState)
SONG_LENGTH = 500
step_Num = 1
for i in range(SONG_LENGTH-1):
    predict_result,dyrnnNew = sess.run([logits,dyrnn],feed_dict={INPT:generate_get, keep_prob:1.0, InitState: state_new, stepNum:step_Num})

    
    predict_result = np.exp(predict_result)
    sm = np.sum(predict_result)
    predict_result /= sm

    predict_result = np.reshape(predict_result, len(predict_result[0]))
  
    index = np.random.choice(range(vecDim),p = predict_result)
    #for i in index:
    songMatrix.append(index)
   
    generate_get = np.reshape(index, [batchSz,step_Num])
    state_new = dyrnnNew[1]
print 'predict_result is~~~~~~~~~~~~~',songMatrix
    #print 'predict size is~~~~~~~~',len(predict_result)
    #print 'predict size is~~~~~~~~',len(predict_result[0])
    #print 'predict_result[0] is~~~~~~~~~~~~~',predict[0]
real_song = []
temp = songMatrix[0]
prevstate = [[0,0] for x in range(span)]
prevstate[temp-1] = [1,1]
real_song.append(prevstate)
for i in range(1, len(songMatrix)):
    print i
    prevstate = [[0,0] for x in range(span)]
    notes = songMatrix[i]
    
    if notes!= 0 and notes == temp:
        prevstate[notes-1] = [1,0]
    elif notes!=0 and notes != temp:
        prevstate[notes-1] = [1,1]
        temp = notes
    real_song.append(prevstate)
S = np.array(real_song)
statematrix = np.hstack((S[:, :, 0], S[:, :, 1]))
statematrix = np.asarray(statematrix).tolist()
end = [0]*68
statematrix.append(end)
noteStateMatrixToMidi(statematrix, "melody-to-melody4")