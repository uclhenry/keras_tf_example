# -*- coding: utf-8 -*-
from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D
from keras.layers import concatenate
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
DATA_DIR = "../data"

MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40

EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 10

# Read training data and generate vocabulary
maxlen = 0
word_freqs = collections.Counter()
num_recs = 0
training_data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
training_data['sentence_q'] = training_data['sentence']
training_data['sentence_a'] = training_data['sentence']
#for line in ftrain:
for line in training_data.iterrows():
    label ,sentence_q = line[1]['label'],line[1]['sentence_q']
    label ,sentence_a = line[1]['label'],line[1]['sentence_a']
    words = nltk.word_tokenize(sentence_q.lower())#.decode("ascii", "ignore")
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:
        word_freqs[word] += 1
    words = nltk.word_tokenize(sentence_a.lower())#.decode("ascii", "ignore")
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:
        word_freqs[word] += 1
    num_recs += 1

## Get some information about our corpus
#print maxlen            # 42
#print len(word_freqs)   # 2313

# 1 is UNK, 0 is PAD
# We take MAX_FEATURES-1 featurs to accound for PAD
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0]: i+2 for i, x in
                enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v:k for k, v in word2index.items()}

# convert sentences to sequences
X_q = np.empty((num_recs, ), dtype=list)
X_a = np.empty((num_recs, ), dtype=list)
y = np.zeros((num_recs, ))
i = 0
#ftrain = open(os.path.join(DATA_DIR, "umich-sentiment-train.txt"), 'rb')
for line in training_data.iterrows():
    label ,sentence_q,sentence_a = line[1]['label'],line[1]['sentence_q'],line[1]['sentence_a']
    #label, sentence = line.strip().split("\t")
    #print(label,sentence)
    words = nltk.word_tokenize(sentence_q.lower())
    seqs = []
    for word in words:
        if word in word2index.keys():
            seqs.append(word2index[word])
        else:
            seqs.append(word2index["UNK"])
    X_q[i] = seqs
    print('add_q')

    words = nltk.word_tokenize(sentence_a.lower())
    seqs = []
    for word in words:
        if word in word2index.keys():
            seqs.append(word2index[word])
        else:
            seqs.append(word2index["UNK"])
    X_a[i] = seqs
    print('add_a')
    # for sentence in [sentence_q,sentence_a]:
    #     words = nltk.word_tokenize(sentence.lower())
    #     seqs = []
    #     for word in words:
    #         if word in word2index.keys():
    #             seqs.append(word2index[word])
    #         else:
    #             seqs.append(word2index["UNK"])
    #     if sentence_q == sentence:
    #         X_q[i] = seqs
    #         print('add_q')
    #     elif sentence_a == sentence:
    #         X_a[i] = seqs
    #         print('add_a')


    y[i] = int(label)
    i += 1
#ftrain.close()

# Pad the sequences (left padded with zeros)
X_a = sequence.pad_sequences(X_a, maxlen=MAX_SENTENCE_LENGTH)
X_q = sequence.pad_sequences(X_q, maxlen=MAX_SENTENCE_LENGTH)
print(X_a[:2])
print(X_q[:2])
X = []
for i in range(len(X_a)):
    concat = [X_q[i],X_a[i]]
    X.append(concat)
print(X[:2])


# Split input into training and test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2,
                                                random_state=42)


#print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)
Xtrain_Q = [e[0] for e in Xtrain]
Xtrain_A = [e[1] for e in Xtrain]
Xtest_Q = [e[0] for e in Xtest]
Xtest_A = [e[1] for e in Xtest]
# Build model
# model = Sequential()
# model.add(Embedding(vocab_size, EMBEDDING_SIZE,
#                     input_length=MAX_SENTENCE_LENGTH))
# model.add(SpatialDropout1D(0.2))
# model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(1))
# model.add(Activation("sigmoid"))
#
# model.compile(loss="binary_crossentropy", optimizer="adam",
#               metrics=["accuracy"])
#
from keras.layers.wrappers import Bidirectional
from keras.layers import Input
from keras.models import Model

inputs_Q = Input(shape=(MAX_SENTENCE_LENGTH,), name="input")
x_embed = Embedding(vocab_size, EMBEDDING_SIZE,input_length=MAX_SENTENCE_LENGTH)(inputs_Q)
inputs_drop = SpatialDropout1D(0.2)(x_embed)
encoded_Q = Bidirectional(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2,name= 'RNN'))(inputs_drop)

inputs_A = Input(shape=(MAX_SENTENCE_LENGTH,), name="input_a")
x_embed = Embedding(vocab_size, EMBEDDING_SIZE,input_length=MAX_SENTENCE_LENGTH)(inputs_A)
inputs_drop = SpatialDropout1D(0.2)(x_embed)
encoded_A = Bidirectional(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2,name= 'RNN'))(inputs_drop)

x = concatenate([encoded_Q, encoded_A])

matching_x = Dense(128)(x)
matching_x = Activation("sigmoid")(matching_x)
polar = Dense(1)(matching_x)
prop = Activation("sigmoid")(polar)

model = Model(inputs=[inputs_Q,inputs_A], outputs=prop)
model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])


history = model.fit([Xtrain_Q,Xtrain_A], ytrain, batch_size=BATCH_SIZE,
                    epochs=NUM_EPOCHS,
                    validation_data=([Xtest_Q,Xtest_A], ytest))

# plot loss and accuracy
plt.subplot(211)
plt.title("Accuracy")
plt.plot(history.history["acc"], color="g", label="Train")
plt.plot(history.history["val_acc"], color="b", label="Validation")
plt.legend(loc="best")

plt.subplot(212)
plt.title("Loss")
plt.plot(history.history["loss"], color="g", label="Train")
plt.plot(history.history["val_loss"], color="b", label="Validation")
plt.legend(loc="best")

plt.tight_layout()
plt.show()

# evaluate
score, acc = model.evaluate([Xtest_Q,Xtest_A], ytest, batch_size=BATCH_SIZE)
print("Test score: %.3f, accuracy: %.3f" % (score, acc))

for i in range(25):
    idx = np.random.randint(len(Xtest_Q))
    #idx2 = np.random.randint(len(Xtest_A))
    xtest_Q = Xtest_Q[idx].reshape(1,40)
    xtest_A = Xtest_A[idx].reshape(1,40)
    ylabel = ytest[idx]
    ypred = model.predict([xtest_Q,xtest_A])[0][0]
    sent_Q = " ".join([index2word[x] for x in xtest_Q[0].tolist() if x != 0])
    sent_A = " ".join([index2word[x] for x in xtest_A[0].tolist() if x != 0])
    print("%.0f\t%d\t%s\t%s" % (ypred, ylabel, sent_Q,sent_A))
