import os
import logging
import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, config, training=False):
        """
        seq_input: a [ T x B x D ] matrix, where T is the time steps in the batch, B is the
            batch size, and D is the amount of dimensions
        """
        if (config.dropout_prob <= 0.0 or config.dropout_prob > 1.0):
            raise Exception("Invalid dropout probability: {}".format(config.dropout_prob))
        if (config.input_dropout_prob <= 0.0 or config.input_dropout_prob > 1.0):
            raise Exception("Invalid input dropout probability: {}".format(config.input_dropout_prob))

        self.config = config
        self.seq_input = tf.placeholder(tf.float32, shape=[config.time_steps, None, config.input_dim])

        initializer = tf.random_uniform_initializer(-0.1, 0.1)
        # setup variables
        with tf.variable_scope("rnnlstm"):
            output_W = tf.get_variable("output_w", [config.hidden_size, config.harmony_dim], initializer=initializer)
            output_b = tf.get_variable("output_b", [config.harmony_dim], initializer=initializer)
            self.lr = tf.constant(config.learning_rate, name="learning_rate")
            self.lr_decay = tf.constant(config.learning_rate_decay, name="learning_rate_decay")

        def create_cell(input_size):
            if config.cell_type == "vanilla":
                cell_class = tf.nn.rnn_cell.BasicRNNCell
            elif config.cell_type == "gru":
                cell_class = tf.nn.rnn_cell.BasicGRUCell
            elif config.cell_type == "lstm":
                cell_class = tf.nn.rnn_cell.BasicLSTMCell
            else:
                raise Exception("Invalid cell type: {}".format(config.cell_type))

            cell = cell_class(config.hidden_size, input_size=input_size)
            if training:
                return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=config.dropout_prob)
            else:
                return cell

        if training:
            self.seq_input_dropout = tf.nn.dropout(self.seq_input, keep_prob=config.input_dropout_prob)
        else:
            self.seq_input_dropout = self.seq_input

        self.cell = tf.nn.rnn_cell.MultiRNNCell(
            [create_cell(config.input_dim)] + [create_cell(config.hidden_size) for i in range(1, config.num_layers)])

        batch_size = tf.shape(self.seq_input_dropout)[1]
        self.initial_state = self.cell.zero_state(batch_size, tf.float32)
        inputs_list = tf.unpack(self.seq_input_dropout)

        # rnn outputs a list of [batch_size x H] outputs
        outputs_list, self.final_state = tf.nn.rnn(self.cell, inputs_list, initial_state=self.initial_state)

        outputs = tf.pack(outputs_list)
        outputs_concat = tf.reshape(outputs, [-1, config.hidden_size])
        logits_concat = tf.matmul(outputs_concat, output_W) + output_b
        logits = tf.reshape(logits_concat, [config.time_steps, -1, config.harmony_dim])

        # probabilities of each note
        self.probs = self.calculate_probs(logits)
        self.loss = self.init_loss(logits, logits_concat)
        self.train_step = tf.train.RMSPropOptimizer(self.lr, decay = self.lr_decay).minimize(self.loss)


    def init_loss(self, outputs, outputs_concat):
        self.seq_targets = tf.placeholder(tf.int64, [self.config.time_steps, None, 2])  # harmony idx, harmony continuity
        batch_size = tf.shape(self.seq_targets)[1]

        with tf.variable_scope("rnnlstm"):
            self.harmony_coeff = tf.constant(self.config.harmony_coeff)

        targets_concat = tf.reshape(self.seq_targets, [-1, 2])
        harmony_idx_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(outputs_concat[:, :-2], targets_concat[:, 0])
        harmony_continuity_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(outputs_concat[:, -2:], targets_concat[:, 1])
        losses = tf.add(self.harmony_coeff * harmony_idx_loss, (1 - self.harmony_coeff) * harmony_continuity_loss)
        return tf.reduce_sum(losses) / self.config.time_steps / tf.to_float(batch_size)


    def calculate_probs(self, logits):
        steps = []
        for t in range(self.config.time_steps):
            harmony_idx_softmax = tf.nn.softmax(logits[t, :, :-2])
            harmony_continuity_softmax = tf.nn.softmax(logits[t, :, -2:])
            steps.append(tf.concat(1, [harmony_idx_softmax, harmony_continuity_softmax]))
        return tf.pack(steps)


    def get_cell_zero_state(self, session, batch_size):
        return session.run(self.cell.zero_state(batch_size, tf.float32))


    # [Data Format Converters]
    def pickle_batch_to_model_xy(self, batch_data):
        """
        input dimension: [batch_size, time_steps, dim]
        """
        # [batch_size, time_steps, dim] to [time_steps, batch_size, dim]
        data = np.swapaxes(batch_data, 0, 1)
        # cur melody + prev harmony -> cur harmony
        data[:, :, :self.config.melody_dim] = np.roll(data[:, :, :self.config.melody_dim], -1, axis=0)
        targets = np.roll(data, -1, axis=0)
        # cutoff final time step
        x = data[:-1, :, :]
        targets = targets[:-1, :, :]
        assert x.shape == targets.shape

        y = np.zeros((targets.shape[0], targets.shape[1], 2), dtype=np.int32)
        y[:, :, 0] = np.argmax(targets[:, :, self.config.melody_dim:-2], axis=2)
        y[:, :, 1] = np.argmax(targets[:, :, -2:], axis=2)

        return x,y




