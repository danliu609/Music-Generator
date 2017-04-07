import os, sys
import argparse
import time
import itertools

import numpy as np
import tensorflow as tf 

import soundfile as sf
import sounddevice as sd   

import yy_pickle_util as pickle_util
import yy_model as model
import yy_mh_util as mh_util
import yy_pickle_util as pickle_util
import yy_midi_util as midi_util



class DefaultConfig(object):
    # graph parameters
    input_dim = None
    melody_dim = None
    harmony_dim = None

    num_layers = 2
    hidden_size = 200
    harmony_coeff = 0.8  #harmony idx 0.8, harmony continuity 0.2
    dropout_prob = 0.5
    input_dropout_prob = 0.8
    cell_type = 'lstm'

    # learning parameters
    time_steps = 1
    batch_size = 1
    learning_rate = 5e-3
    learning_rate_decay = 0.9


def generate(model_path, melody_fn, output_fn):
    pickle_file = "data/Nottingham_train"

    # readin data; set config;
    dataReader = pickle_util.DataReader(pickle_file)
    config = DefaultConfig()
    config.input_dim = dataReader.dim
    config.melody_dim = dataReader.melody_dim
    config.harmony_dim = dataReader.harmony_dim
    model_class = model.Model

    melody_twohot, multiple_note = mh_util.get_melody_twohot(melody_fn, \
        0, pickle_util.MELODY_MIN, pickle_util.MELODY_MAX)
    harmony_twohot = np.zeros((melody_twohot.shape[0]+1,config.harmony_dim), dtype=np.int32)
    # set the first empty chord
    #harmony_twohot[0,dataReader.chord_to_idx[mh_util.NO_CHORD]] = 1
    #harmony_twohot[0,-1] = 1
    
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("model", reuse=None):
            sampling_model = model_class(config)

        saver = tf.train.Saver(tf.all_variables())
        saver.restore(session, model_path)

        state = sampling_model.get_cell_zero_state(session, 1)
        for t in range(melody_twohot.shape[0]):
            x = np.zeros((1,1,config.input_dim))
            x[0,0,:config.melody_dim] = melody_twohot[t,:]
            x[0,0,config.melody_dim:] = harmony_twohot[t,:]

            target_tensors = [sampling_model.probs, sampling_model.final_state]
            feed = {
                sampling_model.initial_state: state,
                sampling_model.seq_input: x,
            }
            probs, state = session.run(fetches=target_tensors, feed_dict=feed)
            probs = probs[0,0,:]
            # chord
            #idx = np.argmax(probs)
            idx =  np.random.choice(config.harmony_dim-2, 1, p=probs[:-2])
            harmony_twohot[t+1, idx] = 1
            # continuity
            if probs[-2] > probs[-1]:
                harmony_twohot[t+1, -2] = 1
            else:
                harmony_twohot[t+1, -1] = 1

        harmony_twohot = harmony_twohot[1:,:]
        assert melody_twohot.shape[0] == harmony_twohot.shape[0]
        dataWriter = pickle_util.DataWriter(pickle_file)
        dataWriter.twohot_mh_to_mid(melody_twohot, harmony_twohot, output_fn)



def prepare_melody_from_midi(midifile, tracki):
    output_fn = midi_util.shift_note(midifile, tracki, \
        pitchLowerBound=pickle_util.MELODY_MIN, pitchUpperBound=pickle_util.MELODY_MAX)
    return output_fn

def prepare_melody_from_vocal():
    os.system("open -a Melodyne vocal.mpd")
    raw_input()
    output_fn = midi_util.shift_note("vocal.mid", 1, \
        pitchLowerBound=pickle_util.MELODY_MIN, pitchUpperBound=pickle_util.MELODY_MAX)
    return output_fn

if __name__ == '__main__':
    model_path = "models/MODEL_THANKSGIVING/yy_model"
    output_fn = "OUT.mid"

    melody_fn = prepare_melody_from_midi("midi/jigs_simple_chords_5.mid", tracki=1)
    #melody_fn = prepare_melody_from_vocal()

    #melody_fn = "midi/jigs_simple_chords_100.mid"
    generate(model_path, melody_fn, output_fn)
    os.system("open " + output_fn)






