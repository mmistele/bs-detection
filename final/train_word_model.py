import numpy as np
from time import time

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform

from embedding import strings_to_word_indices, pretrained_embedding_layer
from models import Word_Model
from utils import read_csv, read_glove_vecs

X_train, Y_train = read_csv('data/train.csv') 
X_dev, Y_dev = read_csv('data/dev.csv') 

maxLen = max(len(max(X_train, key=len).split()), len(max(X_dev, key=len).split()))
print "Max length: %s" % maxLen

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

model = Word_Model((None,), word_to_vec_map, word_to_index)

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
X_train_indices = strings_to_word_indices(X_train, word_to_index, maxLen)

tensorboard = TensorBoard(log_dir="logs/word-model/{}".format(time()))

checkpoint_path = 'checkpoints/word-model-weights.{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(checkpoint_path, period = 5)

model.fit(X_train_indices, Y_train, epochs = 20, batch_size = 6, shuffle=True, callbacks = [tensorboard, checkpoint])

X_dev_indices = strings_to_word_indices(X_dev, word_to_index, max_len = maxLen)
loss, acc = model.evaluate(X_dev_indices, Y_dev)
model.save('word-model-masked-bidir.h5')
print()
print("Dev accuracy = ", acc)
