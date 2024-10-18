# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pandas.plotting import table
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.data import Dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
!pip install -q -U keras-tuner
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Flatten, Bidirectional , BatchNormalization, TimeDistributed, Dropout
import keras_tuner as kt
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder

cfpb_clean = pd.read_csv('/kaggle/input/cfpb-multiclass/cfpb_multiclass.csv')

cfpb_clean['Consumer complaint narrative lemma'] = cfpb_clean['Consumer complaint narrative lemma'].astype(str)

d = {'Credit reporting, credit repair services, or other personal consumer reports ': 0, 'Debt collection': 1, 'Credit card or prepaid card': 2, 'Bank account or service' : 3, 'Mortgage': 4, 'Money transfer, virtual currency, or money service': 5, 'Vehicle loan or lease' : 6, 'Payday loan, title loan, personal loan, or advance loan': 7, 'Student loan' : 8 }
cfpb_clean['Target'] = cfpb_clean['Product'].map(d)
targets = cfpb_clean['Target']
targets = targets.astype('float32')
texts = cfpb_clean['Consumer complaint narrative lemma']

tokenized_texts = [word_tokenize(text.lower()) for text in texts]
word2vec_model = Word2Vec(sentences=tokenized_texts, vector_size=400, window=5, min_count=1, workers=4)
embedding_matrix = np.zeros((len(word2vec_model.wv.key_to_index) + 1, word2vec_model.vector_size))
for word, i in word2vec_model.wv.key_to_index.items():
    embedding_matrix[i] = word2vec_model.wv[word]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Convert text to sequences of integers
sequences = tokenizer.texts_to_sequences(texts)

# Find the maximum sequence length
max_sequence_length = max(len(seq) for seq in sequences)
X_padded = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

print(max_sequence_length)

print(embedding_matrix[154085])

label_encoder = LabelEncoder()
targets = label_encoder.fit_transform(cfpb_clean['Product'])
one_hot_encoded = pd.get_dummies(targets, prefix='class')
one_hot_encoded = one_hot_encoded.astype('float32')
print(one_hot_encoded)

f1_score = tf.keras.metrics.F1Score(average='macro', threshold = 0.5)
def build_model(hp):
    input_ids = Input(shape=(3351,), dtype=tf.float32, name="input_ids")
    hp_learning_rate = hp.Choice('learning_rate',values = [0.05, 0.001, 0.005])
    embeddings = Embedding(input_dim=len(word2vec_model.wv.key_to_index) + 1, output_dim=word2vec_model.vector_size,input_length=3351, weights=[embedding_matrix],trainable=False)(input_ids)
    out = Bidirectional(LSTM(units=128, dropout=0.2, return_sequences=True))(embeddings)
    out = Dropout(0.2)(out)
    out = BatchNormalization()(out)
    out = TimeDistributed(Dense(9,activation = 'softmax'))(out)
    y = out[:, -1, :]
    model = tf.keras.Model(inputs=input_ids, outputs=y)
    optimizer = tf.keras.optimizers.Adam(lr = hp_learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=f1_score)
    return model

tuner = kt.RandomSearch(build_model, objective='val_loss',
max_trials=3, directory='bda128', project_name='BDA594_Project')
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
tuner.search(x = X_padded, y=one_hot_encoded, epochs=4, validation_split=0.4, callbacks=[stop_early] ,batch_size = 64)
best_hps=tuner.get_best_hyperparameters()[0]
print(best_hps.values)

input_ids = Input(shape=(max_sequence_length,), dtype=tf.float32, name="input_ids")
embeddings = Embedding(input_dim=len(word2vec_model.wv.key_to_index) + 1,input_length=max_sequence_length, output_dim=word2vec_model.vector_size, weights=[embedding_matrix], trainable=False) (input_ids)
out = Bidirectional(LSTM(units=128, dropout=0.2, return_sequences=True))(embeddings)
out = Dropout(0.2)(out)
out = BatchNormalization()(out)
out = TimeDistributed(Dense(9,activation = 'softmax'))(out)
y = out[:, -1, :]
model = tf.keras.Model(inputs=input_ids, outputs=out)
model.summary()

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.005), loss = tf.keras.losses.CategoricalCrossentropy(), metrics=f1_score)
final = model.fit(x =X_padded, epochs = 7, batch_size=50, validation_split=0.4, y=one_hot_encoded)
