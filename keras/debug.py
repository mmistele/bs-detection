import numpy as np

from keras.callbacks import TensorBoard
from keras.models import load_model
from keras import backend as K

from embedding import strings_to_character_vecs
from utils import *

# import pdb
# pdb.set_trace()
model = load_model('character_model.h5')

X_trained_on, _ = read_csv('data/train-debug.csv')

X, Y = read_csv('data/test-debug.csv')

maxLen = len(max(X_trained_on, key=len))
counts = get_char_counts_from_csv('data/train-debug.csv') + get_char_counts_from_csv('data/test-debug.csv') 
most_common = counts.most_common()
char_to_index = {x:i for i, (x, _) in enumerate(most_common)}
index_to_char = {i:x for i, (x, _) in enumerate(most_common)}
alphabet_size = len(counts)
char_to_vec_map = {x : index_to_one_hot(i, alphabet_size) for i, (x, _) in enumerate(most_common)}

X = strings_to_character_vecs(X, char_to_index, maxLen, alphabet_size)

print(model.predict(X))

def print_all_outputs():
    inp = model.input # placeholder
    outputs = [layer.output for layer in model.layers]
    functor = K.function([inp] + [K.learning_phase()], outputs)

    test = np.random.random(model.input.shape)[np.newaxis, ...]
    layer_outs = functor([test, 0.])
    print layer_outs



