"""Define the model."""

import tensorflow as tf
import numpy as np


def build_model(mode, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (tf.Tensor) output of the model
    """
    sentence = inputs['sentence']
    sentence_lengths = inputs['sentence_lengths']


    if params.model_version == 'lstm':
        # Get word embeddings for each token in the sentence
        embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
                shape=[params.vocab_size, params.embedding_size]) # where is "embeddings" getting set??
        sentence = tf.nn.embedding_lookup(embeddings, sentence)

        # sentence = tf.Print(sentence, [sentence], summarize=400)
        # sentence = tf.Print(sentence, [sentence.shape], summarize=100)

        # Apply LSTM over the embeddings
 
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
        logits, _  = tf.nn.dynamic_rnn(lstm_cell, sentence, dtype=tf.float32)
        print(logits.shape)

        # Extra layer
        # logits = tf.layers.dense(logits, params.intermediate_vector_size)
        
        # Our code: mask out the pad words and average over the remaining
        # logits = tf.Print(logits, [logits], summarize = 100)
        print(sentence_lengths)

        logits = tf.Print(logits, [sentence_lengths], summarize = 4)

        # mask = tf.sequence_mask(sentence_lengths)
        # logits = tf.boolean_mask(logits, mask)
        # logits = tf.Print(logits, [logits], summarize = 100)
        # print(logits.shape)

        avg = tf.reduce_mean(logits, axis=1) # check that axis
        print(avg.shape)
        output = tf.layers.dense(avg, 1, activation=None)

    else:
        raise NotImplementedError("Unknown model version: {}".format(params.model_version))

    return logits, output


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']
    # sentence_lengths = inputs['sentence_lengths']

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        _, output = build_model(mode, inputs, params)
        predictions = tf.cast(output > 0.0, tf.int32)

    predictions = tf.Print(predictions, [predictions, output], summarize=10)
    labels = tf.Print(labels, [labels], summarize=10)
    # Define loss and accuracy
    output = tf.squeeze(output)
    # labels = tf.reshape(labels, (labels.get_shape()[0], 1))

    # Define loss and accuracy 
    loss = tf.losses.sigmoid_cross_entropy(labels, output)
    # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = output)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
