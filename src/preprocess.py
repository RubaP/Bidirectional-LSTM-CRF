import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import re, string


def clearup(s, chars):
    str = re.sub('[%s]' % chars, '0', s)
    str = re.sub('-+', '-', str)
    return str

def readfile(filename):
    '''
        read file
        return format :
        [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O']]
    '''
    f = open(filename)
    sentences = []
    sentence = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue
        splits = line.split(' ')
        sentence.append([clearup(splits[0], string.digits), splits[1], splits[-1].replace('\n','')])

    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences


def add_chars(sentences):
    for sentence_index, sentence in enumerate(sentences):
        for word_index, word_info in enumerate(sentence):
            chars = [c for c in word_info[0]]
            sentences[sentence_index][word_index] = [word_info[0], chars, word_info[1], word_info[2]]
    return sentences


def get_casing(word):
    casing = []

    num_of_digits = 0
    for char in word:
        if char.isdigit():
            num_of_digits += 1

    num_of_digits_norm = num_of_digits / float(len(word))

    casing.append(1) if word.isdigit() else casing.append(0)
    casing.append(1) if num_of_digits_norm > 0.5 else casing.append(0)
    casing.append(1) if word.islower() else casing.append(0)
    casing.append(1) if word.isupper() else casing.append(0)
    casing.append(1) if word[0].isupper() else casing.append(0)
    casing.append(1) if num_of_digits > 0 else casing.append(0)
    casing.append(1) if word.isalnum() > 0 else casing.append(0)
    casing.append(1) if word.isalpha() > 0 else casing.append(0)
    casing.append(1) if word.find("\'") >= 0 else casing.append(0)
    casing.append(1) if word == "(" or word == ")" else casing.append(0)
    casing.append(1) if len(word) == 1 else casing.append(0)

    return casing


def create_matrices(sentences, word_index, label_index, char_index, pos_tag_index):
    unknown_index = word_index['UNKNOWN_TOKEN']
    dataset = []

    word_count = 0
    unknown_word_count = 0

    for sentence in sentences:
        word_indices = []
        case_indices = []
        char_indices = []
        label_indices = []
        pos_tag_inices = []

        for word, char, pos_tag, label in sentence:
            word_count += 1
            if word in word_index:
                word_idx = word_index[word]
            elif word.lower() in word_index:
                word_idx = word_index[word.lower()]
            else:
                word_idx = unknown_index
                unknown_word_count += 1
            char_idx = []
            for x in char:
                char_idx.append(char_index[x])
            # Get the label and map to int
            word_indices.append(word_idx)
            case_indices.append(get_casing(word))
            char_indices.append(char_idx)
            label_indices.append(label_index[label])
            pos_tag_inices.append(pos_tag_index[pos_tag])

        dataset.append([word_indices, case_indices, char_indices, label_indices, pos_tag_inices])

    return dataset


def padding(chars, length):
    padded_chair = []
    for i in chars:
        padded_chair.append(pad_sequences(i, length, padding='post'))
    return padded_chair


def create_batches(data, batch_size, pos_tag_index):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    def data_generator():
        """
        Generates a batch iterator for a dataset.
        """
        def get_length(data):
            word, case, char, label, pos_tag = data
            return len(word)

        data_size = len(data)
        data.sort(key=lambda x: get_length(x))

        while True:
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X = data[start_index: end_index]
                max_length_word = max(len(max(seq, key=len)) for seq in X)
                yield transform(X, max(2,max_length_word), pos_tag_index)

    return num_batches_per_epoch, data_generator()


def transform(X, max_length_word, pos_tag_index):
    word_input = []
    char_input = []
    case_input = []
    label_input = []
    pos_tag_input = []

    max_length_char = find_max_length_char(X)

    for word, case, char, label, pos_tag in X:
        word_input.append(pad_sequence(word, max_length_word))
        case_input.append(pad_sequence(case, max_length_word, False, True))
        label_input.append(np.eye(9)[pad_sequence(label, max_length_word)])
        pos_tag_input.append(to_categorical(pad_sequence(pos_tag, max_length_word), num_classes=len(pos_tag_index)))
        char_input.append(pad_sequence(char, max_length_word, True))

    return [np.asarray(word_input), np.asarray(case_input), np.asarray(pos_tag_input), np.asarray(padding(char_input, max_length_char))], [np.asarray(label_input), np.flip(np.asarray(label_input), axis=1)]


def find_max_length_char(X):
    max_length = 0;
    for word, case, char, label, pos_tag in X:
        for ch in char:
            if len(ch) > max_length:
                max_length = len(ch)
    return max_length


def pad_sequence(seq, pad_length, isChair = False, isCasing = False):
    if isChair:
        for x in range(len(seq), pad_length):
            seq.append([])
        return seq
    elif isCasing:
        for x in range(pad_length - len(seq)):
            seq.append(np.zeros(11))
        return seq
    else:
        return np.pad(seq, (0, pad_length - len(seq)), 'constant', constant_values=(0,0))


def get_words_and_labels(train, val, test):
    label_set = set()
    pos_tag_set = set()
    words = {}

    for dataset in [train, val, test]:
        for sentence in dataset:
            for word, char, POS_tag, label in sentence:
                label_set.add(label)
                pos_tag_set.add(POS_tag)
                words[word.lower()] = True
    return words, label_set, pos_tag_set
