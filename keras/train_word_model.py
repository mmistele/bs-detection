import numpy as np
from time import time

from keras.callbacks import TensorBoard
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform

from embedding import strings_to_word_indices, pretrained_embedding_layer
from models import Word_Model
from utils import read_csv, read_glove_vecs

X_train, Y_train = read_csv('data/train.csv') 
X_test, Y_test = read_csv('data/test.csv') 

maxLen = len(max(X_train, key=len).split())

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

model = Word_Model((maxLen,), word_to_vec_map, word_to_index)

# might want to change the metric here
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train_indices = strings_to_word_indices(X_train, word_to_index, maxLen)

tensorboard = TensorBoard(log_dir="logs/word-model/{}".format(time()))

model.fit(X_train_indices, Y_train, epochs = 20, batch_size = 6, shuffle=True, callbacks = [tensorboard])

X_test_indices = strings_to_word_indices(X_test, word_to_index, max_len = maxLen)
loss, acc = model.evaluate(X_test_indices, Y_test)
model.save('word-model.h5')
print()
print("Test accuracy = ", acc)
