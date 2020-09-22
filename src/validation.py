from keras.callbacks import Callback
from src.preprocess import transform
import numpy as np

# Method to compute the accuracy. Call predict_labels to get the labels for the dataset
def compute_f1(predictions, correct, idx2Label):
    label_pred = []
    for sentence in predictions:
        label_pred.append([idx2Label[element] for element in sentence])

    label_correct = []
    for sentence in correct:
        label_correct.append([idx2Label[element] for element in sentence])

    prec = compute_precision(label_pred, label_correct)
    rec = compute_precision(label_correct, label_pred)

    f1 = 0
    if (rec + prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec);

    return prec, rec, f1


def compute_precision(guessed_sentences, correct_sentences):
    assert (len(guessed_sentences) == len(correct_sentences))
    correctCount = 0
    count = 0

    for sentenceIdx in range(len(guessed_sentences)):
        guessed = guessed_sentences[sentenceIdx]
        correct = correct_sentences[sentenceIdx]
        assert (len(guessed) == len(correct))
        idx = 0
        while idx < len(guessed):
            if guessed[idx][0] == 'B':  # A new chunk starts
                count += 1

                if guessed[idx] == correct[idx]:
                    idx += 1
                    correctlyFound = True

                    while idx < len(guessed) and guessed[idx][0] == 'I':  # Scan until it no longer starts with I
                        if guessed[idx] != correct[idx]:
                            correctlyFound = False

                        idx += 1

                    if idx < len(guessed):
                        if correct[idx][0] == 'I':  # The chunk in correct was longer
                            correctlyFound = False

                    if correctlyFound:
                        correctCount += 1
                else:
                    idx += 1
            else:
                idx += 1

    precision = 0
    if count > 0:
        precision = float(correctCount) / count

    return precision
    
class Metrics(Callback):
    
    def __init__(self, train_data, idx2Label, pos_tag_index):
        self.valid_data = train_data    
        self.idx2Label = idx2Label
        self.pos_tag_index = pos_tag_index
        
    def on_train_begin(self, logs={}):         
        return
    
    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):        
        return
    
    def on_epoch_end(self, epoch, logs={}):
        dataset = self.valid_data                
        correctLabels = []
        predLabels = []
        for i, data in enumerate(dataset):
            tokens, casing, char, labels, pos_tag = data
            input, output = transform([[tokens, casing, char, labels, pos_tag]], max(2,len(labels)), self.pos_tag_index)
            pred = self.model.predict(input, verbose=False)
            pred = np.add(np.squeeze(pred[0]), np.flip(np.squeeze(pred[1]), axis=0))
            pred = pred.argmax(axis=-1)  # Predict the classes
            output = np.squeeze(output[0])
            output = np.argmax(output, axis=1)
            correctLabels.append(output)
            predLabels.append(pred)                   
        
        pre_dev, rec_dev, f1_dev = compute_f1(predLabels, correctLabels, self.idx2Label)
        print("Dev-Data: Prec: %.5f, Rec: %.5f, F1: %.5f" % (pre_dev, rec_dev, f1_dev))

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return