'''Implement dynamic memory networks based on https://arxiv.org/pdf/1603.01417.pdf'''


import tensorflow as tf
import numpy as np


def position_encoder(sentence_size, embedding_size):
    '''Position encoding is described in section 4.1 in "End to End Memory Networks" in more detail (http://arxiv.org/pdf/1503.08895v5.pdf)'''
    encoding = np.zeros((embedding_size, sentence_size), dtype=np.float32)
    for k in range(1, embedding_size + 1):
        for j in range(1, sentence_size + 1):
            encoding[k-1, j-1] = (1 - j / sentence_size) - (k / embedding_size) * (1 - 2 * j / sentence_size)
    return np.transpose(encoding)


class DMN_plus():
    '''A DMN+ model for textual QA'''
    def __init__(self, hparams, iterator, vocab_dict, reverse_vocab_dict):
        '''Create the model

        Args:
            hparams: hyperparameter configuration
            iterator: Tensorflow dataset iterator that feeds data
            vocab_dict: a dict map token to index
            reverse_vocab_dict: a dict revert index to token

        '''
        self.embedding_size = hparams.embedding_size
        self.sentence_size = hparams.sentence_size
        self.num_of_memory_pass = hparams.num_of_memory_pass

        self.learning_rate = hparams.learning_rate
        self.max_gradient = hparams.max_gradient
        self.dropout_rate = hparams.dropout_rate

        self.iterator = iterator
        self.vocab_dict = vocab_dict
        self.reverse_vocab_dict = reverse_vocab_dict

        self.vocab_size = len(self.vocab_dict)
        self.sentence_encoding = position_encoder(self.sentence_size, self.embedding_size)

        # keep_prob is (1 - self.dropout_rate) for training, and set to 1 for validation & test
        self.keep_prob = tf.placeholder(tf.float32)

        self.logits, self.loss, self.batch_accuracy = self.build_graph()

        self.global_step = tf.Variable(0, trainable=False)
        self.train_op = self.get_train_op()
        self.saver = tf.train.Saver()


    def build_graph(self):
        '''Build the compute graph'''
        inputs, inputs_len, _, questions, questions_len, answers, answers_len = self.iterator.get_next()

        # Word Embedding
        with tf.variable_scope('embedding'):
            embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size], dtype=tf.float32)

        # Input Module
        with tf.variable_scope('textual_input'):
            input_embedding = tf.nn.embedding_lookup(embedding, inputs)
            with tf.variable_scope('sentence_reader'):
                # sentence_vector shape: [batch_size, max_context_len, max_sentence_len, embedding_size]
                sentence_vector = input_embedding * self.sentence_encoding
                # sentence_fact shape: [batch_size, max_context_len, embedding_size]
                sentence_fact = tf.reduce_sum(sentence_vector, 2)
            with tf.variable_scope('fusion_layer'):
                cell_fw = tf.nn.rnn_cell.GRUCell(self.embedding_size)
                cell_bw = tf.nn.rnn_cell.GRUCell(self.embedding_size)
                outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, sentence_fact, sequence_length=inputs_len, dtype=tf.float32)
                # facts shape: [batch_size, max_context_len, embedding_size]
                facts = tf.reduce_sum(tf.stack(outputs), 0)
                facts = tf.nn.dropout(facts, self.keep_prob)

        # Question Module
        with tf.variable_scope('question'):
            question_embedding = tf.nn.embedding_lookup(embedding, questions)
            cell = tf.nn.rnn_cell.GRUCell(self.embedding_size)
            _, question_vector = tf.nn.dynamic_rnn(cell, question_embedding, sequence_length=questions_len, dtype=tf.float32)

        # Episodic Memory Module
        with tf.variable_scope('episodic_memory'):
            # memory^0 is set to question vector
            prev_memory = question_vector
            for i in range(self.num_of_memory_pass):
                attention = self.get_attention(facts, question_vector, prev_memory, 'attention')
                attention = tf.expand_dims(attention, -1)
                # use soft attention instead of attn_gru
                c = tf.reduce_sum(facts * attention, 1)
                # generate episodic memory using ReLU
                with tf.variable_scope('memory_pass_{}'.format(i)):
                    prev_memory = tf.layers.dense(tf.concat([prev_memory, c, question_vector], -1), self.embedding_size, activation=tf.nn.relu)

            memory = tf.nn.dropout(prev_memory, self.keep_prob)

        # Answer Module
        with tf.variable_scope('answer'):
            logits = tf.layers.dense(tf.concat([question_vector, memory], -1), self.vocab_size, activation=None)

        # Compute loss
        labels = tf.squeeze(answers, -1)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        # Statistics
        correct = tf.equal(tf.argmax(logits, axis=-1, output_type=tf.int32), labels)
        batch_accuracy = tf.reduce_sum(tf.cast(correct, tf.int32)) / tf.shape(correct)[0]

        return logits, loss, batch_accuracy


    def get_attention(self, fact, question, memory, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Broadcast question and memory to each fact
            question = tf.expand_dims(question, 1)
            memory = tf.expand_dims(memory, 1)
            z = tf.concat([fact * question, fact * memory, tf.abs(fact - question), tf.abs(fact - memory)], -1)
            z = tf.layers.dense(z, self.embedding_size, activation=tf.nn.tanh, name='dense_0')
            z = tf.layers.dense(z, 1, activation=None, name='dense_1')
            # attention shape: [batch_size, max_context_len]
            g = tf.nn.softmax(tf.squeeze(z, -1))

        return g


    def get_train_op(self):
        params = tf.trainable_variables()
        # No L2 regularation for loss here
        train_loss = self.loss
        gradients = tf.gradients(train_loss, params)
        clipped_gradients = [tf.clip_by_norm(grad, self.max_gradient) for grad in gradients]
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        return train_op
