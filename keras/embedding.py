import numpy as np
from keras.layers.embeddings import Embedding

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
    
    embedding_layer = Embedding(vocab_len, emb_dim, trainable = False) # can also set mask_zero = True

    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix]) # now it's pretrained!

    return embedding_layer

def strings_to_word_indices(X, word_to_index, max_len):
    """
    Converts an array of strings into an array of indices corresponding to words in the string.
    
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
               X_indices[i, j] = 0 # HACK - fix soon
            else:
                X_indices[i, j] = word_to_index[w]
            j = j+1
    return X_indices

def strings_to_character_vecs(X, char_to_index, max_len, alphabet_size):
    """
    Converts an array of strings into an array of one-hot vectors corresponding to characters in the string.
    
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