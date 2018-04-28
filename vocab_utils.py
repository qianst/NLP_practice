'''Functions to preprocess the texts for NLP'''

import re


def generate_vocab(source_file, tokenizer, vocab_file=None):
    '''All the tokens (including punctuation) in the source file will be sorted by the descent of token frequency.

    Args:
        source_file: each line in the source_file is a sentence
        tokenizer: the tokenize function, we need different for English and Chinese
        vocab_file: if vocab_file is not None, the vocab will be saved to vocab_file

    Returns:
        a list of token by frequency descent

    '''
    token_dict = {}
    with open(source_file, 'r') as sentences:
        for sentence in sentences:
            for token in tokenizer(sentence):
                if token in token_dict:
                    token_dict[token] += 1
                else:
                    token_dict[token] = 1

    token_freq_list = sorted(token_dict.items(), key=lambda item: item[1], reverse=True)
    vocabulary = [token for token, freq in token_freq_list]

    if vocab_file is not None:
        with open(vocab_file, 'w') as f:
            for token in vocabulary:
                f.write('{}\n'.format(token))

    return vocabulary


def tokenize(sentence):
    '''Return the tokens of a sentence including punctuation, for languages like English

    >>> tokenize("It's a test.")
    ['It', "'", 's', 'a', 'test', '.']

    '''
    return [token.strip() for token in re.split('(\W+)', sentence) if token.strip()]


def tokenize_cn(sentence):
    '''Return the tokens of a sentence including punctuation, for languages like Chinese'''
    return [token for token in sentence.strip()]


def load_vocab(vocab_file):
    '''Load vocabulary from vocab_file.

    Each line in the vocab_file is a token, sorted by the descent of token frequency.

    '''
    with open(vocab_file, 'r') as f:
        vocab_list = [token.strip() for token in f]
    return vocab_list


def add_special_token_to_vocab(vocab_list, special_tokens=['<unk>', '<s>', '</s>']):
    '''Add special token to vocabulary'''
    return special_tokens + vocab_list


def build_vocab_dict(vocab_list, vocab_max_size=None):
    '''Generate a vocabulary dict from a vocab_list, speeding up lookup

    Args:
        vocab_list: a list of token sorted by the decsent of the token frequency
        max_vocab_size: if max_vocab_size is defined, the first `vocab_max_size` tokens will be used

    Returns:
        vocab_dict: token to index dict
        reverse_vocab_dict: index to token dict

    '''
    if vocab_max_size is not None:
        vocab_list = vocab_list[:vocab_max_size]

    reverse_vocab_dict =dict(enumerate(vocab_list))
    vocab_dict = dict(zip(reverse_vocab_dict.values(), reverse_vocab_dict.keys()))

    return vocab_dict, reverse_vocab_dict


def load_sentences(source_file):
    '''Return a list of sentences, each sentences is a token-split list

    Each line in the source_file is a sentence

    '''
    sentence_list = []
    with open(source_file, 'r') as sentences:
        for sentence in sentences:
            sentence_list.append([token for token in sentence.strip()])
    return sentence_list


def add_sos_prefix(sentence_list, sos_prefix='<s>'):
    '''Add sos prefix to sentences if it is the decoder input during train'''
    return [[sos_prefix] + sentence for sentence in sentence_list]


def add_eos_suffix(sentence_list, eos_suffix='</s>'):
    '''Add eos suffix to sentences if it is the decoder output during train'''
    return [sentence + [eos_suffix] for sentence in sentence_list]


def unify_sentences(sentence_list, sentence_len, sentence_num=None, padding_symbol='</s>'):
    '''Truncate or pad all the sentences to the same length, and add dummy sentences if needed

    Args:
        sentence_list: the list of sentences to unify, each sentence is a string
        sentence_len: the unified sentence length
        sentence_num: the total number of sentences after unified, if it is None, keep len(sentence_list) unchanged
        padding_symbol: default is </s>, means end of sentence

    Returns:
        unified_sentence_list: the unified sentence_list
        original_sentence_lens: a list records the corresponding original sentence length for each sentence in unified_sentence_list
        original_sentence_num: the original number of sentence in the sentence_list before adding dummy sentence

    '''
    # truncation
    if sentence_num is not None:
        sentence_list = sentence_list[-sentence_num:]
    sentence_list = [s[-sentence_len:] for s in sentence_list]

    # padding
    unified_sentence_list = []
    original_sentence_lens = []
    original_sentence_num = len(sentence_list)

    for sentence in sentence_list:
        length = len(sentence)
        if length < sentence_len:
            sentence += [padding_symbol] * (sentence_len - length)
        unified_sentence_list.append(sentence)
        original_sentence_lens.append(length)

    if sentence_num is not None and len(sentence_list) < sentence_num:
        dummy_sentence = [padding_symbol] * sentence_len
        unified_sentence_list += [dummy_sentence] * (sentence_num - len(sentence_list))

    return unified_sentence_list, original_sentence_lens, original_sentence_num


def map_token_to_index(sentences, vocab_dict):
    '''Map each token in each sentence to the index representation by the vocab_dict

    Args:
        sentences: a list of sentence, each sentence is a list of individual tokens
        vocab_dict: the token to index dict

    '''
    # replace out of vocabulary token with <unk>
    unk_index = vocab_dict.get('<unk>')

    sentences_by_index = []
    for sentence in sentences:
        sentences_by_index.append([vocab_dict.get(token, unk_index) for token in sentence])

    return sentences_by_index


def revert_index_to_token(index_list, reverse_vocab_dict):
    '''Revert a single sentence from an index_list

    Args:
        index_list: each element in index_list is an index of a single token from reverse_vocab_dict, the whole list represents a sentence
        reverse_vocab_dict: the index to token dict

    '''
    assert max(index_list) <= len(reverse_vocab_dict)

    return [reverse_vocab_dict[index] for index in index_list]
