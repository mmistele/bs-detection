import numpy as np
from utils import *
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform

X_train, Y_train = read_csv('data/train.csv') 
X_test, Y_test = read_csv('data/test.csv') 

maxLen = len(max(X_train, key=len).split())

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    m = X.shape[0]
    X_indices = np.zeros((m, max_len))
    for i in range(m):
        sentence_words = X[i].lower().split()
        j = 0
        for w in sentence_words:
            if w not in word_to_index:
               X_indices[i, j] = 0 # HACK - FIX SOON
            else:
                X_indices[i, j] = word_to_index[w]
            j = j+1
    return X_indices

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    vocab_len = len(word_to_index) + 1
    emb_dim = word_to_vec_map["lemon"].shape[0]
    emb_matrix = np.zeros((vocab_len, emb_dim)) # curious why not transpose of this...
    # Sets each row "index" of the embedding matrix to be 
    # the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]
    
    embedding_layer = Embedding(vocab_len, emb_dim, trainable = False)

    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix]) # now it's pretrained!

    return embedding_layer

embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

def Model_V1(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the Model-V1 model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """
    
    sentence_indices = Input(shape = input_shape, dtype = np.int32)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    # Propagates sentence_indices through the embedding layer
    embeddings = embedding_layer(sentence_indices)

    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    X = LSTM(128, return_sequences = True)(embeddings)
    # Adds dropout with probability 0.5
    X = Dropout(0.5)(X)
    # Another LSTM layer, but just returns one output
    X = LSTM(128)(X)
    X = Dropout(0.5)(X)
    # Propagating through a Dense layer with sigmoid activation to get back a scalar
    X = Dense(1)(X)
    X = Activation('sigmoid')(X)

    model = Model(inputs = sentence_indices, outputs = X)

    return model

model = Model_V1((maxLen,), word_to_vec_map, word_to_index)

# might want to change the metric here
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)

model.fit(X_train_indices, Y_train, epochs = 20, batch_size = 6, shuffle=True)

X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
loss, acc = model.evaluate(X_test_indices, Y_test)
model.save('my_model.h5')
print()
print("Test accuracy = ", acc)
