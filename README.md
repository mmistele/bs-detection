# Gibberfish
Final project for CS 230: Deep Learning by Matt Mistele & Bo Peng

What if language models were trained not by having them try to predict the next word, but predict whether the phrase or sentence as a whole was syntactically correct and made sense? If done right, the activation of the hidden state neurons in a trained RNN might capture different aspects of how human written sentence are structured, without any explicit part of speech tagging or other labeling more granular than a 0 or a 1 for the sequence as a whole.

We were interested to see whether something akin to part of speech representations would emerge in the hidden states as a result of training for a task in which they would be useful, perhaps essential. So, we trained RNNs to predict whether a sequence of words is a well-formed English sentence or a sequence of words randomly sampled from the corpus, using the 50,000 sentence Tatoeba corpus for positive examples and randomly saampled word sequences from the corpus for negative examples.

Our 2-layer word LSTM with GloVe word embeddings is 96% accurate at distinguishing valid sentences from nonsensical ones. The analysis of what was learned in the hidden states is ongoing, with preliminary visualizations in the Weight Analysis notebook.
