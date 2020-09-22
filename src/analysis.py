
def print_wrong_tags(sentence_info, prediction, idx2Label):
    print("Total sentences: ", len(sentence_info))
    wrong_words = 0

    for sentence_index, sentence in enumerate(sentence_info):
        if len(sentence) == len(prediction[sentence_index]):
            for word_index, word_info in enumerate(sentence):
                if word_info[3] != "O":
                    predicted = idx2Label[prediction[sentence_index][word_index]]
                    if word_info[3] != predicted:
                        wrong_words +=1
                        print(wrong_words, " : ", "Word:", word_info[0], "  Original:", word_info[3] +"  Predicted:",
                              idx2Label[prediction[sentence_index][word_index]])
