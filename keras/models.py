import numpy as np

from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Masking, Bidirectional, Lambda

from custom_layers import MeanPool

from embedding import pretrained_embedding_layer

def Character_Model_1(input_shape):
    """
    Function creating the Character Model-V1 model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """    
    sentences = Input(shape = input_shape, dtype = np.float32)

    X = Masking(mask_value = 0., input_shape=input_shape)(sentences)

    # X = Bidirectional(LSTM(128, return_sequences = False, dropout=0.2, recurrent_dropout=0.2), merge_mode='ave')(X)
    X = LSTM(128, return_sequences = True, dropout=0.2, recurrent_dropout=0.2, name='LSTM1')(X)
    def get_last(X):
        return X[:,-1,:]
    X = Lambda(get_last, name = 'LSTM-last')(X)

    # X = Bidirectional(LSTM(128, return_sequences = True, dropout=0.2, recurrent_dropout=0.2))(X)
    # X = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(X)
    
    X = Dense(1, name='Dense1')(X)
    X = Activation('sigmoid', name='output_layer')(X)

    model = Model(inputs = sentences, outputs = X)
    return model

def Character_Model_2(input_shape):
    sentences = Input(shape = input_shape, dtype = np.float32)

    X = Masking(mask_value = 0., input_shape=input_shape)(sentences)
    X = LSTM(128, return_sequences = True, dropout=0.2, recurrent_dropout=0.2)(X)
    X = LSTM(128, return_sequences = True, dropout=0.2, recurrent_dropout=0.2)(X)
    # X = Masking(mask_value = 0., input_shape=(input_shape[0], 128))(X)

    X = MeanPool()(X)
    X = Dense(1)(X)
    X = Activation('sigmoid')(X)

    model = Model(inputs = sentences, outputs = X)
    return model


def Word_Model(input_shape, word_to_vec_map, word_to_index):
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