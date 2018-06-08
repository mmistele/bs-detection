import numpy as np
from time import time

from keras.callbacks import TensorBoard
from keras.models import Model, load_model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform

from embedding import strings_to_word_indices, pretrained_embedding_layer
from models import Word_Model
from utils import read_csv, read_glove_vecs
model = load_model('super-good-word-model.h5')

X_trained_on, _ = read_csv('data/train-big.csv')
X_dev, Y_dev = read_csv('data/dev.csv')
X_test, Y_test = read_csv('data/test_minus_dev_big.csv')

maxLen = max(len(max(X_trained_on, key=len).split()), len(max(X_dev, key=len).split()))+10
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
X_test_indices = strings_to_word_indices(X_test, word_to_index, maxLen)

loss, acc = model.evaluate(X_test_indices, Y_test)
print()
print("Test accuracy = ", acc)