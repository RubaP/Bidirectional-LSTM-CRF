import numpy as np


def get_word_embedding(words):
    word_index = {}
    word_embeddings = []
    embeddings = open("../embedding/glove.6B.100d.txt", encoding="utf-8")

    for line in embeddings:
        split = line.strip().split(" ")

        if len(word_index) == 0:  # Add padding+unknown
            word_index["PADDING_TOKEN"] = len(word_index)
            vector = np.zeros(len(split) - 1)  # Zero vector vor 'PADDING' word
            word_embeddings.append(vector)

            word_index["UNKNOWN_TOKEN"] = len(word_index)
            vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
            word_embeddings.append(vector)

        if split[0].lower() in words:
            vector = np.array([float(num) for num in split[1:]])
            word_embeddings.append(vector)
            word_index[split[0]] = len(word_index)

    word_embeddings = np.array(word_embeddings)
    return word_index, word_embeddings


def get_char_index_matrix():
    char_index = {"PADDING": 0, "UNKNOWN": 1}
    for c in " 0abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
        char_index[c] = len(char_index)
    return char_index


def get_label_index_matrix():
    label_index = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3, 'I-LOC': 4, 'B-ORG': 5, 'I-ORG': 6, 'B-MISC': 7,
                   'I-MISC': 8}
    return label_index


def get_pos_index_matrix(POS_tag_set):
    POS_tag_index = {"PAD":0}
    for POS_tag in POS_tag_set:
        POS_tag_index[POS_tag] = len(POS_tag_index)
    return POS_tag_index
