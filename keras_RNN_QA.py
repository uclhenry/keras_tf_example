# -*- coding: utf-8 -*-
from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D

from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import jieba
from alime_data import convert_dialogue_to_pair
from parameter import MAX_SENTENCE_LENGTH,MAX_FEATURES,embedding_size,max_pair,batch_size,HIDDEN_LAYER_SIZE,NUM_EPOCHS
def get_huabei(name):
    huabei_data = pd.read_csv(name,delimiter='\t')
    #print(huabei_data.info())
    N = 10000
    n = 0
    data = []
    for line in huabei_data.itertuples():
        a,b,Question,candidate,label = line
        #print(line,len(line))
        #print(Question,'-----',candidate,label)
        #print
        n +=1
        data.append([Question,candidate,label])
        # if n > N:
        #     break
        #break
    return pd.DataFrame(data,columns=['sentence_q','sentence_a','label'])
def downsample_huabei():
    training_data1 = get_huabei('atec_nlp_sim_train.csv')
    training_data2 = get_huabei('atec_nlp_sim_train_add.csv')
    training_data = pd.concat([training_data1, training_data2], axis=0)
    training_data_positive = training_data[training_data['label'] == 1]
    training_data_negative = training_data[training_data['label'] == 0]
    number_positive, number_negative = len(training_data_positive), len(training_data_negative)
    training_data = pd.concat([training_data_negative.sample(number_positive), training_data_positive], axis=0)
    return training_data
def upsample_huabei():
    training_data1 = get_huabei('atec_nlp_sim_train.csv')
    training_data2 = get_huabei('atec_nlp_sim_train_add.csv')
    training_data = pd.concat([training_data1, training_data2], axis=0)
    training_data_positive = training_data[training_data['label'] == 1]
    training_data_negative = training_data[training_data['label'] == 0]
    number_positive, number_negative = len(training_data_positive), len(training_data_negative)
    sizes = []
    while number_negative > 0:
        if number_negative > number_positive:
            sizes.append(number_positive)
        else:
            sizes.append(number_negative)
        number_negative -= number_positive
    #print(sizes)

    samples = [training_data_positive.sample(size) for size in sizes]
    return  pd.concat([training_data_negative]+samples,axis=0)

DATA_DIR = "../data"

# Read training data and generate vocabulary
maxlen = 0
num_recs = 0

word_freqs = collections.Counter()
#training_data = convert_dialogue_to_pair(max_pair)


training_data = upsample_huabei()
count_label = pd.value_counts(training_data['label'].values)
print(count_label)
#print(count_label['0'],count_label['1'])
num_recs  = len([1 for r in training_data.iterrows()])
def chinese_split(x):
    return  list(jieba.cut(x))
#for line in ftrain:
for line in training_data.iterrows():
    label ,sentence_q = line[1]['label'],line[1]['sentence_q']
    label ,sentence_a = line[1]['label'],line[1]['sentence_a']
    #words = nltk.word_tokenize(sentence_q.lower())#.decode("ascii", "ignore")
    words = chinese_split(sentence_q)
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:
        word_freqs[word] += 1
    #words = nltk.word_tokenize(sentence_a.lower())#.decode("ascii", "ignore")
    words = chinese_split(sentence_a)
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:
        word_freqs[word] += 1
    #num_recs += 1
## Get some information about our corpus

# 1 is UNK, 0 is PAD
# We take MAX_FEATURES-1 featurs to accound for PAD
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v:k for k, v in word2index.items()}
# convert sentences to sequences
X_q = np.empty((num_recs, ), dtype=list)
X_a = np.empty((num_recs, ), dtype=list)
y = np.zeros((num_recs, ))
i = 0
# def chinese_split(x):
#     return x.split(' ')

for line in training_data.iterrows():
    label ,sentence_q,sentence_a = line[1]['label'],line[1]['sentence_q'],line[1]['sentence_a']
    #label, sentence = line.strip().split("\t")
    #print(label,sentence)
    #words = nltk.word_tokenize(sentence_q.lower())
    words = chinese_split(sentence_q)
    seqs = []
    for word in words:
        if word in word2index.keys():
            seqs.append(word2index[word])
        else:
            seqs.append(word2index["UNK"])
    X_q[i] = seqs
    #print('add_q')
    #words = nltk.word_tokenize(sentence_a.lower())
    words = chinese_split(sentence_a)
    seqs = []
    for word in words:
        if word in word2index.keys():
            seqs.append(word2index[word])
        else:
            seqs.append(word2index["UNK"])
    X_a[i] = seqs
    y[i] = int(label)
    i += 1
# Pad the sequences (left padded with zeros)
X_a = sequence.pad_sequences(X_a, maxlen=MAX_SENTENCE_LENGTH)
X_q = sequence.pad_sequences(X_q, maxlen=MAX_SENTENCE_LENGTH)
X = []
for i in range(len(X_a)):
    concat = [X_q[i],X_a[i]]
    X.append(concat)

# Split input into training and test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1,
                                                random_state=42)

#print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)
Xtrain_Q = [e[0] for e in Xtrain]
Xtrain_A = [e[1] for e in Xtrain]
Xtest_Q = [e[0] for e in Xtest]
Xtest_A = [e[1] for e in Xtest]

from keras.layers.wrappers import Bidirectional
from keras.layers import Input,Lambda
from keras.models import Model

def encoder(inputs_seqs,rnn_hidden_size,dropout_rate):
    x_embed = Embedding(vocab_size, embedding_size, input_length=MAX_SENTENCE_LENGTH)(inputs_seqs)
    inputs_drop = SpatialDropout1D(0.2)(x_embed)
    encoded_Q = Bidirectional(
        LSTM(rnn_hidden_size, dropout=dropout_rate, recurrent_dropout=dropout_rate, name='RNN'))(inputs_drop)
    return encoded_Q

# a = [a1,a2,a3],b = [b1,b2,b3] ,return [|a1-b1|,|a2-b2|,|a3-b3|]
def absolute_difference(vecs):
    a,b =vecs
    #d = a-b
    return abs(a - b)

def product(vecs):
    a,b =vecs
    #d = a-b
    return a * b
def model_rnn_qa_matching():
    inputs_Q = Input(shape=(MAX_SENTENCE_LENGTH,), name="input")
    inputs_A = Input(shape=(MAX_SENTENCE_LENGTH,), name="input_a")
    encoded_Q = encoder(inputs_Q,HIDDEN_LAYER_SIZE,0.1)
    encoded_A = encoder(inputs_A,HIDDEN_LAYER_SIZE,0.1)
    similarity  =  Lambda(absolute_difference)([encoded_Q, encoded_A])
    # x = concatenate([encoded_Q, encoded_A])
    #
    # matching_x = Dense(128)(x)
    # matching_x = Activation("sigmoid")(matching_x)
    polar = Dense(1)(similarity)
    prop = Activation("sigmoid")(polar)
    model = Model(inputs=[inputs_Q,inputs_A], outputs=prop)
    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=["accuracy"])
    return model
# plot loss and accuracy
def plot(training_history,title):
    plt.subplot(211)
    plt.title(title+" Accuracy")
    plt.plot(training_history.history["acc"], color="g", label="Train")
    plt.plot(training_history.history["val_acc"], color="b", label="Validation")
    plt.legend(loc="best")

    plt.subplot(212)
    plt.title(title+" Loss")
    plt.plot(training_history.history["loss"], color="g", label="Train")
    plt.plot(training_history.history["val_loss"], color="b", label="Validation")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


model = model_rnn_qa_matching()
# sum = count_label[1] + count_label[0]
# weight0,weight1 = count_label[1]/sum , count_label[0]/sum
training_history = model.fit([Xtrain_Q, Xtrain_A], ytrain, batch_size=batch_size,
                             epochs=NUM_EPOCHS,
                             validation_data=([Xtest_Q,Xtest_A], ytest))#,class_weight={0: weight0, 1: weight1}

plot(training_history,'_'.join([str(max_pair),str(vocab_size)]))
# evaluate
score, acc = model.evaluate([Xtest_Q,Xtest_A], ytest, batch_size = batch_size)
print("Test score: %.3f, accuracy: %.3f" % (score, acc))
ylabels = []
ypreds = []
def get_q(idx):
    return Xtest_Q[idx].reshape(1, MAX_SENTENCE_LENGTH)


def get_a(idx):
    return Xtest_A[idx].reshape(1, MAX_SENTENCE_LENGTH)


def get_predict(model,xtest_Q,xtest_A):
    ypred = model.predict([xtest_Q, xtest_A])[0][0]
    if ypred>=0.5:ypred = 1
    return ypred
for i in range(25):
    idx = np.random.randint(len(Xtest_Q))
    #idx2 = np.random.randint(len(Xtest_A))
    xtest_Q = Xtest_Q[idx].reshape(1,MAX_SENTENCE_LENGTH)
    xtest_A = Xtest_A[idx].reshape(1,MAX_SENTENCE_LENGTH)
    ylabel = ytest[idx]
    #ypred = model.predict([xtest_Q,xtest_A])[0][0]
    ypred = get_predict(model,xtest_Q,xtest_A)
    ylabels.append(ylabel)
    ypreds.append(ypred)
    #print(type(ylabel),type(ypred))
    sent_Q = " ".join([index2word[x] for x in xtest_Q[0].tolist() if x != 0])
    sent_A = " ".join([index2word[x] for x in xtest_A[0].tolist() if x != 0])
    #print("%.0f\t%d\t%s\t%s" % (ypred, ylabel, sent_Q,sent_A))

from sklearn.metrics import classification_report

test_size = 100
ylabels = [int(ytest[idx])  for idx in range(test_size)]

# print(y_true)
ypreds = [get_predict(model,get_q(idx),get_a(idx))for idx in range(test_size)]
def getint(x):
    if x >0.5:
        return 1
    return 0
# print(y_pred)
# i = 0
# print([len(x) for x in ylabels])
# print([len(x) for x in ypreds])
ylabels = [int(y) for y in ylabels]
ypreds = [getint(y) for y in ypreds]
def check_label(ylabels,ypreds):
    print(len([y for y in ylabels if y ==1 ]))
    print(len([y for y in ylabels if y ==0 ]))
    print(len([y for y in ypreds if y ==1 ]))
    print(len([y for y in ypreds if y ==0 ]))
print(classification_report(ylabels, ypreds))
