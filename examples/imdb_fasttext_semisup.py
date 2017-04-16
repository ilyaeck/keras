'''This example demonstrates the use of fasttext for text classification

Based on Joulin et al's paper:

Bags of Tricks for Efficient Text Classification
https://arxiv.org/abs/1607.01759

Results on IMDB datasets with uni and bi-gram embeddings:
    Uni-gram: 0.8813 test accuracy after 5 epochs. 8s/epoch on i7 cpu.
    Bi-gram : 0.9056 test accuracy after 5 epochs. 2s/epoch on GTX 980M gpu.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import os
import sys

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.datasets import imdb
from keras import backend as K




TEXT_DATA_DIR = '/Users/IE/Copy/data/movies/sentiment/aclImdb'
GLOVE_VECS_PATH = '/Users/IE/Copy/data/stanford-glove/glove.twitter.27B.50d.txt'
EMBEDDING_DIMS = 50
MAX_SEQUENCE_LENGTH = 400
MAX_FEATURES = 20000
VALIDATION_SPLIT = 0.2


def load_labeled_data(datadir, tokenizer=None):
    print('Processing text dataset in {}'.format(datadir))
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    for name in sorted(os.listdir(datadir)):
        if name == 'unsup':
            continue
        path = os.path.join(datadir, name)
        if os.path.isdir(path): # each label corresponds to a separate directory
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname[0].isdigit():
                    fpath = os.path.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='latin-1')
                    texts.append(f.read())
                    f.close()
                    labels.append(label_id)

    print('Found {} texts in {}.'.format(len(texts), datadir))

    # finally, vectorize the text samples into a 2D integer tensor
    if not tokenizer:
        tokenizer = Tokenizer(nb_words=MAX_FEATURES)
        tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels =  np.asarray(labels) # to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # randomize order
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    return data, labels, tokenizer


def load_train_test_data(train_test_separate=False):
    if train_test_separate: # labeled data is already split into train and test directories
         train_data, train_labels, tokenizer = load_labeled_data(TEXT_DATA_DIR + '/train/')
         test_data, test_labels, _ = load_labeled_data(TEXT_DATA_DIR + '/test/', tokenizer)
    else:
         data, labels, tokenizer = load_labeled_data(TEXT_DATA_DIR)
         nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
         train_data = data[:-nb_validation_samples]
         train_labels = labels[:-nb_validation_samples]
         test_data = data[-nb_validation_samples:]
         test_labels = labels[-nb_validation_samples:]
    return (train_data, train_labels), (test_data, test_labels), tokenizer.word_index

def load_word_vecs():
    print('Indexing word vectors.')
    embeddings_index = {}
    f = open(GLOVE_VECS_PATH)
    iline = 0
    for line in f:
        values = line.split()
        if 0 == iline:
            nvecs = int(values[0])
            ndims = int(values[1])
            if ndims != EMBEDDING_DIMS:
                raise AssertionError("Pretrained vectors must have a matching dimension!")
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        iline += 1
    f.close()
    return embeddings_index




def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.

    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list)-ngram_range+1):
            for ngram_value in range(2, ngram_range+1):
                ngram = tuple(new_list[i:i+ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


def make_embedding_matrix(word_index):
    embeddings_index = load_word_vecs()
    print('Found %s word vectors.' % len(embeddings_index))
    nb_words = min(MAX_FEATURES, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIMS))
    for word, i in word_index.items():
        if i > MAX_FEATURES:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix, nb_words

# Set parameters:
# ngram_range = 2 will add bi-grams features
ngram_range = 1
maxlen = 400
batch_size = 32
nb_epoch = 8
use_pretrained_wvecs = False

print('Loading data...')
(X_train, y_train), (X_test, y_test), word_index = load_train_test_data(train_test_separate=True)
#(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=MAX_FEATURES)

#print('Using  the top {} words from data'.format(nb_words))

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
print('Average train sequence length: {}'.format(np.mean(list(map(len, X_train)), dtype=int)))
print('Average test sequence length: {}'.format(np.mean(list(map(len, X_test)), dtype=int)))


if ngram_range > 1:
    print('Adding {}-gram features'.format(ngram_range))
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in X_train:
        for i in range(2, ngram_range+1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than MAX_FEATURES in order
    # to avoid collision with existing features.
    start_index = MAX_FEATURES + 1
    token_indice = {v: k+start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # MAX_FEATURES is the highest integer that could be found in the dataset.
    MAX_FEATURES = np.max(list(indice_token.keys())) + 1

    # Augmenting X_train and X_test with n-grams features
    X_train = add_ngram(X_train, token_indice, ngram_range)
    X_test = add_ngram(X_test, token_indice, ngram_range)
    print('Average train sequence length: {}'.format(np.mean(list(map(len, X_train)), dtype=int)))
    print('Average test sequence length: {}'.format(np.mean(list(map(len, X_test)), dtype=int)))

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)


# prepare embedding matrix
embedding_matrix = None
if use_pretrained_wvecs:
    embedding_matrix, nb_words = make_embedding_matrix(word_index)

print('Build model...')
model = Sequential()

if embedding_matrix is None:
    # we start off with an efficient embedding layer which maps
    # our vocab indices into EMBEDDING_DIMS dimensions
    model.add(Embedding(MAX_FEATURES,
                        EMBEDDING_DIMS,
                        input_length=maxlen))
else:
    # load pre-trained word embeddings into an Embedding layer
    model.add(Embedding(nb_words + 1,
                        EMBEDDING_DIMS,
                        weights=[embedding_matrix],
                        input_length=maxlen,
                        trainable=True)) # use pre-trained vecs for initial state only

# we add a GlobalAveragePooling1D, which will average the embeddings
# of all words in the document
model.add(GlobalAveragePooling1D())

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))
