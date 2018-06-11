# Gibberfish
Final project for CS 230: Deep Learning by Matt Mistele & Bo Peng

We trained RNNs to predict whether a sequence of words is a well-formed English sentence or a sequence of words randomly sampled from the corpus. 

Part of the idea was to see if the activation of the hidden state neurons in the trained model would capture different aspects of how human written sentence are structured. In particular, we were interested to see whether something akin to part of speech representations would emerge in the hidden states as a result of training for a task in which they would be useful, perhaps essential. 

Our 2-layer word LSTM with GloVe word embeddings is 96% accurate at distinguishing valid sentences from nonsensical ones. The analysis of what was learned in the hidden states is ongoing, with preliminary visualizations in the Weight Analysis notebook.
