import numpy as np
from utils import *
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras.utils import to_categorical

X_train, Y_train = read_csv('data/train-debug.csv') 
X_test, Y_test = read_csv('data/test.csv') 

maxLen = len(max(X_train, key=len))

counts = get_char_counts_from_csv('data/train-debug.csv') + get_char_counts_from_csv('data/test.csv') 
most_common = counts.most_common()
char_to_index = {x:i for i, (x, _) in enumerate(most_common)}
index_to_char = {i:x for i, (x, _) in enumerate(most_common)}
alphabet_size = len(counts)
char_to_vec_map = {x : index_to_one_hot(i, alphabet_size) for i, (x, _) in enumerate(most_common)}

def sentences_to_vec(X, char_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to characters in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    m = X.shape[0]

    # sequences = list()
    # for i in range(m):
    #     line = X[i].lower()
    #     encoded_seq = [char_to_index[char] for char in line]
    #     sequences.append(encoded_seq)
    # X_vec = np.array([to_categorical(seq, num_classes=alphabet_size) for seq in sequences])
    # import pdb; pdb.set_trace()
    # return X_vec


    X_vec = np.zeros((m, max_len, alphabet_size))
    for i in range(m):
        line = X[i].lower()
        j = 0
        for c in line:
            if c in char_to_index:
                X_vec[i, j, char_to_index[c]] = 1.
            j = j+1
    return X_vec


def Model_V1(input_shape):
    """
    Function creating the Model-V1 model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """
    
    sentences = Input(shape = input_shape, dtype = np.float32)
    # Propagate the sentences through an LSTM layer with 128-dimensional hidden state
    X = LSTM(128, return_sequences = True)(sentences)
    # Adds dropout with probability 0.5
    X = Dropout(0.5)(X)

    X = LSTM(128, return_sequences = True)(X)
    X = Dropout(0.9)(X)

    # Another LSTM layer, but just returns one output
    X = LSTM(128)(X)
    X = Dropout(0.5)(X)
    # Propagating through a Dense layer with sigmoid activation to get back a scalar
    X = Dense(1)(X)
    X = Activation('sigmoid')(X)

    model = Model(inputs = sentences, outputs = X)

    return model

# m by maxLen by alphabet_size
X_train_indices = sentences_to_vec(X_train, char_to_index, maxLen)
print(X_train_indices.shape)

model = Model_V1((X_train_indices.shape[1], X_train_indices.shape[2]))

# might want to change the metric here
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train_indices, Y_train, epochs = 20, batch_size = 6, shuffle=True)

X_test_indices = sentences_to_vec(X_test, char_to_index, max_len = maxLen)
loss, acc = model.evaluate(X_test_indices, Y_test)
model.save('character_model.h5')
print()
print("Test accuracy = ", acc)
