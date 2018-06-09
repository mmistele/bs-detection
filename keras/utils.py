import csv
import numpy as np
from collections import Counter

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix

# From emo_utils.py
def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
        
        UNK = len(words)
        index_to_words[UNK] = 'UNK'
        words_to_index['UNK'] = UNK 
        word_to_vec_map['UNK'] = np.random.randn(50) # embedding dimension

    return words_to_index, index_to_words, word_to_vec_map

def read_csv(filename):
    sentences = []
    labels = []

    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            sentences.append(row[0])
            labels.append(row[1])

    X = np.asarray(sentences)
    Y = np.asarray(labels, dtype=int)

    return X, Y

def get_char_counts_from_csv(filename):
    counts = Counter()

    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            counts += Counter(row[0].lower())
    return counts

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def index_to_one_hot(idx, length):
    vec = np.zeros((length, 1))
    vec[idx] = 1
    return vec

def label_to_judgment(label):
    return "real" if int(label) == 1 else "fake"

def print_predictions(X, pred):
    print()
    for i in range(X.shape[0]):
        print(X[i], label_to_judgment(pred[i]))

# def plot_confusion_matrix(y_actu, y_pred, title='Confusion matrix', cmap=plt.cm.gray_r):
    
#     df_confusion = pd.crosstab(y_actu, y_pred.reshape(y_pred.shape[0],), rownames=['Actual'], colnames=['Predicted'], margins=True)
    
#     df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    
#     plt.matshow(df_confusion, cmap=cmap) # imshow
#     #plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(df_confusion.columns))
#     plt.xticks(tick_marks, df_confusion.columns, rotation=45)
#     plt.yticks(tick_marks, df_confusion.index)
#     #plt.tight_layout()
#     plt.ylabel(df_confusion.index.name)
#     plt.xlabel(df_confusion.columns.name)


