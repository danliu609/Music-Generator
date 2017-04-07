import os, sys
import argparse
import time
import itertools
import cPickle
import logging
import random
import string

import numpy as np
import tensorflow as tf    
import matplotlib.pyplot as plt

import yy_pickle_util as pickle_util
import yy_model as model



class DefaultConfig(object):
    # graph parameters
    input_dim = None
    melody_dim = None
    harmony_dim = None

    num_layers = 1  # mean how many extra cell; can be set to 0, which means only one cell, no extra;
    hidden_size = 200
    harmony_coeff = 0.8  #harmony idx 0.8, harmony continuity 0.2
    dropout_prob = 0.5
    input_dropout_prob = 0.8
    cell_type = 'lstm'

    # learning parameters
    time_steps = 128
    batch_size = 64
    learning_rate = 5e-3
    learning_rate_decay = 0.9
    num_epochs = 10000

    def __str__(self):
        return "[Config] input_dim: {}, melody_dim: {}, harmony_dim: {}\n".format(self.input_dim, self.melody_dim, self.harmony_dim) + \
            "num_layers: {}, hidden_size: {}, harmony_coeff: {}, dropout_prob: {}, input_dropout_prob: {}, cell_type: {}\n".format( \
                self.num_layers, self.hidden_size, self.harmony_coeff, self.dropout_prob, self.input_dropout_prob, self.cell_type) + \
            "time_steps: {} , batch_size: {}, learning_rate: {}, learning_rate_decay: {}, num_epochs: {}\n".format( \
                self.time_steps, self.batch_size, self.learning_rate, self.learning_rate_decay, self.num_epochs)


def train(run_folder):
    train_pickle = "data/Nottingham_train"
    valid_pickle = "data/Nottingham_valid"

    # set up run dir
    while os.path.exists(run_folder):
        run_folder += "_0"
    os.makedirs(run_folder)

    logger = logging.getLogger(__name__) 
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(os.path.join(run_folder, "training.log")))

    # readin data; set up config;
    train_data = pickle_util.DataReader(train_pickle)
    valid_data = pickle_util.DataReader(valid_pickle)
    config = DefaultConfig()
    config.input_dim = train_data.dim
    config.melody_dim = train_data.melody_dim
    config.harmony_dim = train_data.harmony_dim
    model_class = model.Model
    all_valid_batches = valid_data.get_all_batches(config.time_steps + 1, config.batch_size)

    logger.info(config)

    best_valid_perplexity = np.inf
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("model", reuse=None):
            train_model = model_class(config, training=True)
        with tf.variable_scope("model", reuse=True):
            valid_model = model_class(config, training=False)

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=10)
        tf.initialize_all_variables().run()

        # training
        want_to_save = False
        state = train_model.get_cell_zero_state(session, config.batch_size) 
        for i in range(config.num_epochs):
            data = train_data.get_batch_random(config.time_steps + 1, config.batch_size)
            x,y = train_model.pickle_batch_to_model_xy(data)

            target_tensors = [train_model.loss, train_model.final_state, train_model.train_step]
            feed_dict = {
                train_model.initial_state: state,
                train_model.seq_input: x,
                train_model.seq_targets: y,
            }

            train_loss, state, _ = session.run(fetches=target_tensors, feed_dict=feed_dict)
            train_perplexity = np.exp(train_loss)
            logger.info("Epoch {}: train_loss is {}, train_perplexity is {})".format(i, train_loss, train_perplexity))
            
            T = 200
            if i % T == (T-1):  # try to save after every T iterations;
                want_to_save = True

            if want_to_save:
                valid_loss_sum = 0.0
                valid_state = state
                # do validation: cover the whole valid_data
                for data in all_valid_batches:
                    x,y = valid_model.pickle_batch_to_model_xy(data)
                    feed_dict = {
                        valid_model.initial_state: valid_state,
                        valid_model.seq_input: x,
                        valid_model.seq_targets: y,
                    }
                    valid_loss, valid_state = session.run(fetches=[valid_model.loss, valid_model.final_state], feed_dict=feed_dict)
                    valid_loss_sum += valid_loss

                valid_loss = valid_loss_sum/len(all_valid_batches)
                valid_perplexity = np.exp(valid_loss)
                logger.info("Valid: valid_loss is {}, valid_perplexity is {})".format(valid_loss, valid_perplexity))
            
                if valid_perplexity < best_valid_perplexity:
                    want_to_save = False
                    best_valid_perplexity = valid_perplexity
                    logger.info("New best_valid_perplexity. Saving model...".format(best_valid_perplexity))
                    saver.save(session, os.path.join(run_folder, "yy_model"))



def test(model_path):
    test_pickle = "data/Nottingham_test"

    # readin data; set config;
    test_data = pickle_util.DataReader(test_pickle)
    config = DefaultConfig()
    config.input_dim = test_data.dim
    config.melody_dim = test_data.melody_dim
    config.harmony_dim = test_data.harmony_dim
    model_class = model.Model
    all_test_batches = test_data.get_all_batches(config.time_steps + 1, config.batch_size)

    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("model", reuse=None):
            test_model = model_class(config, training=False)

        saver = tf.train.Saver(tf.all_variables())
        saver.restore(session, model_path)

        # testing
        test_state = test_model.get_cell_zero_state(session, config.batch_size) 
        test_loss_sum = 0.0
        # do validation: cover the whole test_data
        for data in all_test_batches:
            x,y = test_model.pickle_batch_to_model_xy(data)

            feed_dict = {
                test_model.initial_state: test_state,
                test_model.seq_input: x,
                test_model.seq_targets: y,
            }
            test_loss, test_state = session.run(fetches=[test_model.loss, test_model.final_state], feed_dict=feed_dict)
            test_loss_sum += test_loss

        test_loss = test_loss_sum/len(all_test_batches)
        test_perplexity = np.exp(test_loss)
        print "Test_loss is {}, test_perplexity is {})".format(test_loss, test_perplexity)




if __name__ == '__main__':
    run_folder = "models/MODEL_DEC01"
    train(run_folder)

    # MODEL_THANKSGIVING: Test_loss is 0.0954094661607, test_perplexity is 1.10010922041)
    #model_path = "models/MODEL_THANKSGIVING/yy_model"
    #test(model_path)



