import numpy as np
from utils import read_csv, read_glove_vecs, get_char_counts_from_csv, index_to_one_hot
from embedding import strings_to_character_vecs, strings_to_word_indices
import argparse

parser = argparse.ArgumentParser(description='Load and test a model.')
parser.add_argument('model', metavar='M', nargs=1,
                    help='the model filename (or path)')
parser.add_argument('--legacy', dest='isLegacy', action='store_const',
                    const=True, default=False,
                    help='whether to use legacy word embeddings')
parser.add_argument('--isCharacterModel', dest='isCharacterModel', action='store_const',
                    const=True, default=False,
                    help='whether the word model is a character model')

args = parser.parse_args()

model = load_model(args.model)
X_test, Y_test = read_csv('data/test.csv')

# For the purposes of maxLen calculation
x_trained_on_filename = 'data/train-big.csv'
x_dev_used_filename = 'data/dev.csv'
X_trained_on, _ = read_csv(x_trained_on_filename)
X_dev, Y_dev = read_csv(x_dev_used_filename)

if args.isCharacterModel:
    maxLen = max(len(max(X_trained_on, key=len)), len(max(X_dev, key=len)))
    counts = get_char_counts_from_csv('data/train-big.csv') + get_char_counts_from_csv('data/dev.csv') 
    most_common = counts.most_common()
    char_to_index = {x:i for i, (x, _) in enumerate(most_common)}
    index_to_char = {i:x for i, (x, _) in enumerate(most_common)}
    alphabet_size = len(counts)
    char_to_vec_map = {x : index_to_one_hot(i, alphabet_size) for i, (x, _) in enumerate(most_common)}
    X_test_indices = strings_to_character_vecs(X_test, char_to_index, maxLen, alphabet_size)

else:
    maxLen = max(len(max(X_trained_on, key=len).split()), len(max(X_dev, key=len).split()))+10
    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
    if args.isLegacy:
        word_to_index, index_to_word, word_to_vec_map = read_glove_vecs_legacy('data/glove.6B.50d.txt')
    X_test_indices = strings_to_word_indices(X_test, word_to_index, maxLen, args.isLegacy)

loss, acc = model.evaluate(X_test_indices, Y_test)
print()
print("Test accuracy = ", acc)