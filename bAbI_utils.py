'''bAbI utilities for DMN'''

from collections import Counter

import vocab_utils


def gen_corpus(source_file):
    '''Generate a corpus contain tuples of (context, question, answer)'''
    corpus = []
    with open(source_file, 'r') as f:
        for line in f:
            line_num, sentence = line.strip().split(' ', 1)
            if line_num == '1':
                line_set = []
            if '\t' in sentence:
                context = line_set[:]
                question, answer, _ = sentence.split('\t')
                corpus.append((context, question.strip(), answer.strip()))
            else:
                line_set.append(sentence)

    return corpus


def gen_vocab(corpus):
    '''Generate vocab dict from corpus'''
    tokens= []

    for context, question, answer in corpus:
        context_token = []
        for x in context:
            context_token += vocab_utils.tokenize(x)
        question_token = vocab_utils.tokenize(question)
        answer_token = vocab_utils.tokenize(answer)
        tokens += context_token + question_token + answer_token

    vocab = Counter(tokens)
    vocab = sorted(vocab.items(), key=lambda item: item[1], reverse=True)
    vocab = [item[0] for item in vocab]
    vocab = vocab_utils.add_special_token_to_vocab(vocab)
    vocab_dict, reverse_vocab_dict = vocab_utils.build_vocab_dict(vocab)

    return vocab_dict, reverse_vocab_dict


def get_corpus_shape(corpus):
    '''Get the max_context_len, max_sentence_len, max_question_len of the corpus'''
    contexts = []
    questions = []
    answers = []

    for context, question, answer in corpus:
        context = [vocab_utils.tokenize(sentence) for sentence in context]
        question = vocab_utils.tokenize(question)
        answer = vocab_utils.tokenize(answer)
        contexts.append(context)
        questions.append(question)
        answers.append(answer)

    max_context_len = max(len(context) for context in contexts)
    max_sentence_len = max(len(sentence) for sentence in context for context in contexts)
    max_question_len = max(len(sentence) for sentence in questions)
    max_answer_len = max(len(answer) for answer in answers)

    return max_context_len, max_sentence_len, max_question_len, max_answer_len


def preprocess(corpus, vocab_dict, max_context_len, max_sentence_len, max_question_len, max_answer_len):
    '''Tokenize corpus, map to index and pad each sentence to the same length'''
    contexts = []
    questions = []
    answers = []

    for context, question, answer in corpus:
        context = [vocab_utils.tokenize(sentence) for sentence in context]
        question = vocab_utils.tokenize(question)
        answer = vocab_utils.tokenize(answer)
        contexts.append(context)
        questions.append(question)
        answers.append(answer)

    unified_contexts = []
    unified_contexts_len = []
    unified_sentence_len = []

    for context in contexts:
        context, sentence_len, sentence_num = vocab_utils.unify_sentences(context, max_sentence_len, max_context_len, padding_symbol='</s>')
        context = vocab_utils.map_token_to_index(context, vocab_dict)
        unified_contexts.append(context)
        unified_contexts_len.append(sentence_num)
        unified_sentence_len.append(sentence_len)
    # pad list of sentence_len to length of context_len with 0
    unified_sentence_len, _, _ = vocab_utils.unify_sentences(unified_sentence_len, max_context_len, padding_symbol=0)

    questions, questions_len, _ = vocab_utils.unify_sentences(questions, max_question_len, padding_symbol='</s>')
    questions = vocab_utils.map_token_to_index(questions, vocab_dict)

    answers, answers_len, _ = vocab_utils.unify_sentences(answers, max_answer_len, padding_symbol='</s>')
    answers = vocab_utils.map_token_to_index(answers, vocab_dict)

    return unified_contexts, unified_contexts_len, unified_sentence_len, questions, questions_len, answers, answers_len
