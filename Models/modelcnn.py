import pandas as pd
import os
import gzip
import argparse
import time
import re
import jieba
import pickle
import tensorflow as tf
import numpy as np
import sys, getopt
from subprocess import check_output
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import SeparableConv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import GlobalAveragePooling1D
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from tensorflow.python.keras.preprocessing import text
from tensorflow.python.keras.preprocessing import sequence


from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


MAX_SEQ_LENGTH = 500
MAX_VOCAB_SIZE = 20000 # Limit on the number of features. We use the top 20K features

# code form https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
def clean_en_text(dat):
    
    REPLACE_BY_SPACE_RE = re.compile('["/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')
    
    ret = []
    for line in dat:
        # text = text.lower() # lowercase text
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        line = BAD_SYMBOLS_RE.sub('', line)
        line = line.strip()
        ret.append(line)
    return ret

def clean_zh_text(dat):
    REPLACE_BY_SPACE_RE = re.compile('[“”【】/（）：！～「」、|，；。"/(){}\[\]\|@,\.;]')
    
    ret = []
    for line in dat:
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        line = line.strip()
        ret.append(line)
    return ret



def sequentialize_data(train_contents, val_contents=None):
    """Vectorize data into ngram vectors.

    Args:
        train_contents: training instances
        val_contents: validation instances
        y_train: labels of train data.

    Returns:
        sparse ngram vectors of train, valid text inputs.
    """
    tokenizer = text.Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(train_contents)
    x_train = tokenizer.texts_to_sequences(train_contents)

    if val_contents:
        x_val = tokenizer.texts_to_sequences(val_contents)

    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQ_LENGTH:
        max_length = MAX_SEQ_LENGTH

    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    if val_contents:
        x_val = sequence.pad_sequences(x_val, maxlen=max_length)

    word_index = tokenizer.word_index
    num_features = min(len(word_index) + 1, MAX_VOCAB_SIZE)
    if val_contents:
        return x_train, x_val, word_index, num_features, tokenizer, max_length
    else:
        return x_train, word_index, num_features, tokenizer, max_length


def _get_last_layer_units_and_activation(num_classes):
    """Gets the # units and activation function for the last network layer.

    Args:
        num_classes: Number of classes.

    Returns:
        units, activation values.
    """
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation


def sep_cnn_model(input_shape,
                  num_classes,
                  num_features,
                  embedding_matrix,
                  blocks=1,
                  filters=64,
                  kernel_size=4,
                  dropout_rate=0.5):
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)

    model = models.Sequential()
    model.add(Embedding(input_dim=num_features, output_dim=300, input_length=input_shape,
                        embeddings_initializer=Constant(embedding_matrix)))

    for _ in range(blocks - 1):
        model.add(Dropout(rate=dropout_rate))
        model.add(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(MaxPooling1D(pool_size=3))

    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))

    model.add(GlobalAveragePooling1D())
    # model.add(MaxPooling1D())
    model.add(Dropout(rate=0.5))
    model.add(Dense(op_units, activation=op_activation))
    return model


def _tokenize_chinese_words(text):
    return ' '.join(jieba.cut(text, cut_all=False))


def vectorize_data(x_train, x_val=None):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    if x_val:
        full_text = x_train + x_val
    else:
        full_text = x_train
    vectorizer.fit(full_text)
    train_vectorized = vectorizer.transform(x_train)
    if x_val:
        val_vectorized = vectorizer.transform(x_val)
        return train_vectorized, val_vectorized, vectorizer
    return train_vectorized, vectorizer


# onhot encode to category
def ohe2cat(label):
    return np.argmax(label, axis=1)


class Model(object):
    """ 
        model of CNN baseline without pretraining.
        see `https://aclweb.org/anthology/D14-1181` for more information.
    """

    def __init__(self, metadata, train_output_path="./", test_input_path="./"):
        """ Initialization for model
        :param metadata: a dict formed like:
            {"class_num": 10,
             "language": ZH,
             "num_train_instances": 10000,
             "num_test_instances": 1000,
             "time_budget": 300}
        """
        self.done_training = False
        self.metadata = metadata
        self.train_output_path = train_output_path
        self.test_input_path = test_input_path

    def train(self, train_dataset, remaining_time_budget=None):
        """model training on train_dataset.
        
        :param train_dataset: tuple, (x_train, y_train)
            x_train: list of str, input training sentences.
            y_train: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        :param remaining_time_budget:
        """
        if self.done_training:
            return
        x_train, y_train = train_dataset

        # tokenize Chinese words
        if self.metadata['language'] == 'ZH':
            x_train = clean_zh_text(x_train)
            x_train = list(map(_tokenize_chinese_words, x_train))
        else:
            x_train = clean_en_text(x_train)

        x_train, word_index, num_features, tokenizer, max_length = sequentialize_data(x_train)
        num_classes = self.metadata['class_num']

        # loading pretrained embedding
        FT_DIR = 'Embedding'
        fasttext_embeddings_index = {}
        if self.metadata['language'] == 'ZH':
            f = gzip.open(os.path.join(FT_DIR, 'cc.zh.300.vec.gz'),'rb')
        elif self.metadata['language'] == 'EN':
            f = gzip.open(os.path.join(FT_DIR, 'cc.en.300.vec.gz'),'rb')
        else:
            raise ValueError('Unexpected embedding path:'
                             ' {unexpected_embedding}. '.format(
                unexpected_embedding=FT_DIR))

        for line in f.readlines():
            values = line.strip().split()
            if self.metadata['language'] == 'ZH':
                word = values[0].decode('utf8')
            else:
                word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            fasttext_embeddings_index[word] = coefs

        print('Found %s fastText word vectors.' % len(fasttext_embeddings_index))

        # embedding lookup
        EMBEDDING_DIM = 300
        embedding_matrix = np.zeros((num_features, EMBEDDING_DIM))
        cnt = 0
        for word, i in word_index.items():
            if i >= num_features:
                continue
            embedding_vector = fasttext_embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            else:
                embedding_matrix[i] = np.zeros(300)
                cnt += 1

        print ('fastText oov words: %s' % cnt)

        # initialize model
        model = sep_cnn_model(input_shape=x_train.shape[1:][0],
                              num_classes=num_classes,
                              num_features=num_features,
                              embedding_matrix=embedding_matrix,
                              blocks=2,
                              filters=64,
                              kernel_size=4,
                              dropout_rate=0.5)
        if num_classes == 2:
            loss = 'binary_crossentropy'
        else:
            loss = 'sparse_categorical_crossentropy'
        optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10)]

        x_train, y_train = shuffle(x_train, y_train)
        # fit model
        history = model.fit(
            x_train,
            ohe2cat(y_train),
            # y_train,
            epochs=1000,
            callbacks=callbacks,
            validation_split=0.2,
            # validation_data=(x_dev,y_dev),
            verbose=2,  # Logs once per epoch.
            batch_size=32,
            shuffle=True)
        print(str(type(x_train)) + " " + str(y_train.shape))

        # save model
        model.save(self.train_output_path + 'model.h5')
        with open(self.train_output_path + 'tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.train_output_path + 'model.config', 'wb') as f:
            f.write(str(max_length).encode())
            f.close()

        self.done_training = True

    def test(self, x_test, remaining_time_budget=None):
        """
        :param x_test: list of str, input test sentences.
        :param remaining_time_budget:
        :return: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                 here `sample_count` is the number of examples in this dataset as test
                 set and `class_num` is the same as the class_num in metadata. The
                 values should be binary or in the interval [0,1].
        """
        model = models.load_model(self.test_input_path + 'model.h5')
        with open(self.test_input_path + 'tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle, encoding='iso-8859-1')
        with open(self.test_input_path + 'model.config', 'r') as f:
            max_length = int(f.read().strip())
            f.close()

        train_num, test_num = self.metadata['train_num'], self.metadata['test_num']
        class_num = self.metadata['class_num']

        # tokenizing Chinese words
        if self.metadata['language'] == 'ZH':
            x_test = clean_zh_text(x_test)
            x_test = list(map(_tokenize_chinese_words, x_test))
        else:
            x_test = clean_en_text(x_test)

        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = sequence.pad_sequences(x_test, maxlen=max_length)
        result = model.predict_classes(x_test)

        # category class list to sparse class list of lists
        y_test = np.zeros([test_num, class_num])
        for idx, y in enumerate(result):
            y_test[idx][y] = 1
        return y_test

