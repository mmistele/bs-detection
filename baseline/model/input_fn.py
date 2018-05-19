"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf


# TODO: Check this still works for labels file even though it's just a number instead of words
def load_dataset_from_text(path_txt, vocab):
    """Create tf.data Instance from txt file

    Args:
        path_txt: (string) path containing one example per line
        vocab: (tf.lookuptable)

    Returns:
        dataset: (tf.Dataset) yielding list of ids of tokens for each example
    """
    # Load txt file, one example per line
    dataset = tf.data.TextLineDataset(path_txt)

    # Convert line into list of tokens, splitting by white space
    dataset = dataset.map(lambda string: tf.string_split([string]).values)

    # Lookup tokens to return their ids
    dataset = dataset.map(lambda tokens: (vocab.lookup(tokens), tf.size(tokens)))

    return dataset

# We made this one
def load_labels(path_txt):
    """Create tf.data Instance from txt file

    Args:
        path_txt: (string) path containing one label per line

    Returns:
        dataset: (tf.Dataset) yielding value for each example
    """
    # Load txt file, one example per line
    # with open(path_txt) as f:
    #     array = []
    #     for line in f:
    #         array.append([int(x) for x in line.split()[0]])

    dataset = tf.data.TextLineDataset(path_txt)
    dataset = dataset.map(lambda string: tf.cast(string, tf.int32))
    return dataset


def input_fn(mode, sentences, labels, params):
    """Input function for NER

    Args:
        mode: (string) 'train', 'eval' or any other mode you can think of
                     At training, we shuffle the data and have multiple epochs
        sentences: (tf.Dataset) yielding list of ids of words
        datasets: (tf.Dataset) yielding list of ids of tags <- TODO: fix this
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)

    """
    # Load all the dataset in memory for shuffling is training
    is_training = (mode == 'train')
    buffer_size = params.buffer_size if is_training else 1

    # Zip the sentence and the labels together
    dataset = tf.data.Dataset.zip((sentences, labels))
    # import pdb; pdb.set_trace()

    # Create batches and pad the sentences of different length
    # TODO: figure out how to change this to doing many-to-one instead of many-to-many
#     padded_shapes = ((tf.TensorShape([None]),  # sentence of unknown size
#                       tf.TensorShape([])),     # size(words)
#                      (tf.TensorShape([]),      # want this to not pad at all
#                       tf.TensorShape([])))     # size(tags)

    padded_shapes = (
        (tf.TensorShape([None]),  # sentence of unknown size
                      tf.TensorShape([])
        ),     # size(words)
                     tf.TensorShape([])
                     )     # size(tags)
                     #(((tf.TensorShape([None]), tf.TensorShape([])), tf.TensorShape([])))

#     padding_values = ((params.id_pad_word,   # sentence padded on the right with id_pad_word
#                        0),                   # size(words) -- unused
#                       (params.id_pad_word,   # should be unused, making it this for now
#                        0))                   # size(tags) -- unused

#    padding_values = ((params.id_pad_word,   # sentence padded on the right with id_pad_word
#                        0))                   # size(tags) -- unused

    # import pdb; pdb.set_trace()
    dataset = (dataset
        .shuffle(buffer_size=buffer_size)
        .padded_batch(params.batch_size, padded_shapes=padded_shapes, padding_values=((params.id_pad_word, 0), 0))
        .prefetch(1)  # make sure you always have one batch ready to serve
    )

    # Create initializable iterator from this dataset so that we can reset at each epoch
    iterator = dataset.make_initializable_iterator()

    # Query the output of the iterator for input to the model
    ((sentence, sentence_lengths), (labels, _)) = iterator.get_next()
    init_op = iterator.initializer

    # Build and return a dictionnary containing the nodes / ops
    inputs = {
        'sentence': sentence,
        'labels': labels,
        'sentence_lengths': sentence_lengths,
        'iterator_init_op': init_op
    }

    return inputs
