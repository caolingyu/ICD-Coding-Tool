import threading
import sys
sys.path.append('/Users/Leu/ICD-Coding/caml-mimic/')
import datasets
import evaluation
from dataproc import extract_wvs

import numpy as np
import sklearn.metrics as sklm
import tensorflow as tf
import csv
from tqdm import tqdm

from keras.preprocessing import sequence
from keras.models import *
from keras.layers import Input, merge
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.layers import LSTM
from keras.callbacks import Callback
from keras import backend as K


# Evaluation Metrics
def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*((p*r)/(p+r))

def auc(y_true, y_pred):
     auc = tf.metrics.auc(y_true, y_pred)[1]
     K.get_session().run(tf.local_variables_initializer())
     return auc

# Set parameters:
maxlen = 200
embedding_dims = 200
nb_filter = 500
filter_length = 4
batch_size = 8
nb_epoch = 10
nb_labels = 50
train_data_path = "../mimicdata/mimic3/train_50.csv"
dev_data_path = "../mimicdata/mimic3/dev_50.csv"
test_data_path = "../mimicdata/mimic3/test_50.csv"
vocab = "../mimicdata/mimic3/vocab.csv"
embed_file = "../mimicdata/mimic3/processed_full.embed"
dicts = datasets.load_lookups(train_data_path, vocab, Y=nb_labels)
vocab_size = len(dicts[0])
embed_weight = extract_wvs.load_embeddings(embed_file)

# Load data
print('Loading data...')
def slim_data_generator(data_path):
    while 1:
        for batch_idx, tup in enumerate(datasets.data_generator(data_path, dicts, batch_size=batch_size, num_labels=nb_labels)):
            X, y, _, code_set, descs = tup
            X = sequence.pad_sequences(X, maxlen=maxlen)
            yield X, y
            
# class slim_data_generator:
#     def __init__(self, data_path):
#         self.data_path = data_path
#         self.lock = threading.Lock()

#     def __iter__(self):
#         return self

#     def __next__(self):
#         with self.lock:
#             for batch_idx, tup in enumerate(datasets.data_generator(self.data_path, dicts, batch_size=batch_size, num_labels=nb_labels)):
#                 X, y, _, code_set, descs = tup
#                 X = sequence.pad_sequences(X, maxlen=maxlen)
#                 return X, y

gen_slim_train = slim_data_generator(train_data_path)
gen_slim_test = slim_data_generator(test_data_path)
test_y = []
for batch_idx, tup in enumerate(datasets.data_generator(test_data_path, dicts, batch_size=batch_size, num_labels=nb_labels)):
    data, target, hadm_ids, _, descs = tup
    test_y.append(target)
test_y = np.concatenate(test_y, axis=0)

# Helper functions
def count_samples(file_path):
    num_lines = 0
    with open(file_path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            num_lines += 1
        return num_lines
train_samples = count_samples(train_data_path)
test_samples = count_samples(test_data_path)

class evaluate_per_epoch(Callback):
    # def __init__(self, gen_slim_test, test_samples, test_y):
    #     self.gen_slim_test = gen_slim_test
    #     self.test_samples = test_samples
    #     self.test_y = test_y

    def on_epoch_end(self, epoch, logs=None):
        yhat_raw = model.predict_generator(gen_slim_test, steps=test_samples)
        yhat = np.round(yhat_raw)
        #get metrics
        k = 5
        metrics = evaluation.all_metrics(yhat=yhat, y=test_y, k=k, yhat_raw=yhat_raw)
        evaluation.print_metrics(metrics)

evaluate = evaluate_per_epoch()

# Build model
print('Building model...')
inputs = Input(shape=(maxlen,))
# embed = Embedding(vocab_size+2, embedding_dims, input_length=maxlen)(inputs)
embed = Embedding(len(embed_weight),
                  100,
                  weights=[embed_weight],
                  input_length=maxlen,
                  trainable=False)(inputs)
dp1 = Dropout(0.2)(embed)
conv = Convolution1D(nb_filter=nb_filter, filter_length=filter_length, padding='same')(dp1)
alpha = Dense(nb_filter, use_bias=False, activation='softmax', name='attention_vec')(conv)
attention_mul = merge([conv, alpha], output_shape=nb_filter, name='attention_mul', mode='mul')
flat = Flatten()(attention_mul)
output = Dense(nb_labels, activation='sigmoid')(flat)

model = Model(input=[inputs], output=output)
print(model.summary())
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[precision, recall, f1, auc])

# Train model
print('Training model')
model.fit_generator(gen_slim_train, steps_per_epoch=train_samples/batch_size, epochs=nb_epoch,
                    validation_data=gen_slim_test, validation_steps=test_samples/batch_size,
                    callbacks=[evaluate])
