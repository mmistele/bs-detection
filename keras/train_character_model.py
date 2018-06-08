import numpy as np
from time import time

from keras.callbacks import TensorBoard
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras.utils import to_categorical

from embedding import strings_to_character_vecs
from models import Character_Model_1, Character_Model_2
from utils import read_csv, get_char_counts_from_csv, index_to_one_hot


X_train, Y_train = read_csv('data/train.csv') 
X_dev, Y_dev = read_csv('data/dev.csv') 

maxLen = max(len(max(X_train, key=len)), len(max(X_dev, key=len)))
print "Max length: %s" % maxLen

counts = get_char_counts_from_csv('data/train.csv') + get_char_counts_from_csv('data/dev.csv') 
most_common = counts.most_common()
char_to_index = {x:i for i, (x, _) in enumerate(most_common)}
index_to_char = {i:x for i, (x, _) in enumerate(most_common)}
alphabet_size = len(counts)
char_to_vec_map = {x : index_to_one_hot(i, alphabet_size) for i, (x, _) in enumerate(most_common)}

# m by maxLen by alphabet_size
X_train_indices = strings_to_character_vecs(X_train, char_to_index, maxLen, alphabet_size)
print(X_train_indices.shape)

model = Character_Model_1((None, X_train_indices.shape[2]))

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0, epsilon=None)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
tensorboard = TensorBoard(log_dir="logs/character-model/{}".format(time()))
model.fit(X_train_indices, Y_train, epochs = 20, batch_size = 20, shuffle=True, callbacks = [tensorboard], validation_split = 0.2)

X_dev_indices = strings_to_character_vecs(X_dev, char_to_index, maxLen, alphabet_size)
loss, acc = model.evaluate(X_dev_indices, Y_dev)
model.save('character_model.h5')
print()
print("Dev accuracy = ", acc)
