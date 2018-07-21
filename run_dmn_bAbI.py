import collections

import tensorflow as tf

import bAbI_utils
import dmn_plus


def run():

    train_file = r'E:\Downloads\tf_training_data\tasks_1-20_v1-2\en-10k\qa2_two-supporting-facts_train.txt'
    test_file = r'E:\Downloads\tf_training_data\tasks_1-20_v1-2\en-10k\qa2_two-supporting-facts_test.txt'
    out_dir = r'E:\Downloads\tf_training_data\babi_ckpt'

    train_corpus = bAbI_utils.gen_corpus(train_file)
    test_corpus = bAbI_utils.gen_corpus(test_file)

    vocab_dict, reverse_vocab_dict = bAbI_utils.gen_vocab(train_corpus)

    max_context_len, max_sentence_len, max_question_len, max_answer_len = bAbI_utils.get_corpus_shape(train_corpus)
    print('max_context_len = {}, max_sentence_len = {}, max_question_len = {}, max_answer_len = {}'.format(max_context_len, max_sentence_len, max_question_len, max_answer_len))

    batch_size = 100
    epoch = 20

    train_dataset = tf.data.Dataset.from_tensor_slices(bAbI_utils.preprocess(train_corpus, vocab_dict, max_context_len, max_sentence_len, max_question_len, max_answer_len))
    train_dataset = train_dataset.repeat(epoch)
    train_dataset = train_dataset.batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices(bAbI_utils.preprocess(test_corpus, vocab_dict, max_context_len, max_sentence_len, max_question_len, max_answer_len))
    test_dataset = test_dataset.batch(batch_size)

    data_iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = data_iter.make_initializer(train_dataset)
    test_init_op = data_iter.make_initializer(test_dataset)

    keep_prob = tf.placeholder(tf.float32)

    Hparams = collections.namedtuple('Hparams', ['embedding_size', 'sentence_size', 'num_of_memory_pass', 'learning_rate', 'max_gradient', 'dropout_rate'])
    hparams = Hparams(embedding_size=64, sentence_size=max_sentence_len, num_of_memory_pass=3, learning_rate=0.001, max_gradient=10, dropout_rate=0.3)

    model = dmn_plus.DMN_plus(hparams, data_iter, vocab_dict, reverse_vocab_dict)

    logits, loss, batch_accuracy = model.logits, model.loss, model.batch_accuracy
    global_step = model.global_step
    train_op = model.train_op

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(out_dir)
        if ckpt is not None:
            print('Restore model from', ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Training a new model')
            sess.run(tf.global_variables_initializer())

        sess.run(train_init_op)
        while True:
            try:
                res = sess.run([global_step, train_op, loss, logits, batch_accuracy], feed_dict={model.keep_prob:0.7})
                if res[0] % 100 == 0:
                    path_prefix = model.saver.save(sess, out_dir+'\qa_model', global_step=res[0])
                    print('Global step {}: loss = {}, batch_accuracy = {}'.format(res[0], res[2], res[4]))
            except tf.errors.OutOfRangeError:
                print('Training Done')
                break

        sess.run(test_init_op)
        cnt = 0
        correct = 0
        while True:
            try:
                res = sess.run([global_step, loss, logits, batch_accuracy], feed_dict={model.keep_prob:1})
                # print('Global step {}: loss = {}'.format(res[0], res[1]))
                correct += batch_size * res[3]
                cnt += batch_size
            except tf.errors.OutOfRangeError:
                break
        print('Test: correct = {}, total = {}, acc = {}'.format(correct, cnt, correct/cnt))

run()