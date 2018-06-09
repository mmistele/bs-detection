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


model = load_model('word_2lay_128dim_0.5drop_1dir_unmasked_99trainacc.h5')

X_trained_on, _ = read_csv('data/train-big.csv')
X_dev, Y_dev = read_csv('data/dev.csv')
X_test, Y_test = read_csv('data/test.csv')

# Word version
maxLen = max(len(max(X_trained_on, key=len).split()), len(max(X_dev, key=len).split()))+10
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
X_test_indices = strings_to_word_indices(X_test, word_to_index, maxLen)

# Character version
# maxLen = max(len(max(X_trained_on, key=len)), len(max(X_dev, key=len)))
# counts = get_char_counts_from_csv('data/train.csv') + get_char_counts_from_csv('data/dev.csv') 
# most_common = counts.most_common()
# char_to_index = {x:i for i, (x, _) in enumerate(most_common)}
# index_to_char = {i:x for i, (x, _) in enumerate(most_common)}
# alphabet_size = len(counts)
# char_to_vec_map = {x : index_to_one_hot(i, alphabet_size) for i, (x, _) in enumerate(most_common)}
# X_test_indices = strings_to_character_vecs(X_test, char_to_index, maxLen, alphabet_size)

loss, acc = model.evaluate(X_test_indices, Y_test)
print()
print("Test accuracy = ", acc)