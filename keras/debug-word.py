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


model = load_model('word_1lay_64cell_30drop_1dir_60ep_50kex.h5')

X_trained_on, _ = read_csv('data/train-big.csv')
X_dev, Y_dev = read_csv('data/dev.csv')

X, Y = read_csv('data/debug.csv')

# X_test, Y_test = read_csv('data/test_minus_dev_big.csv')

# Word version
maxLen = 33
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
X = strings_to_word_indices(X, word_to_index, maxLen)

print(model.predict(X))