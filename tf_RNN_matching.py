import tensorflow as tf
import nltk
import pandas as pd
from collections import Counter
import numpy as np
import pandas as pd

import time

from parameter import max_pair



def get_pair(number, dialogue):
    pairs = []
    for conversation in dialogue:
        utterances = conversation[2:].strip('\n').split('\t')
        # print(utterances)
        # break

        for i, utterance in enumerate(utterances):
            if i % 2 != 0: continue
            pairs.append([utterances[i], utterances[i + 1]])
            if len(pairs) >= max_pair:
                return pairs
    return pairs


def convert_dialogue_to_pair():
    dialogue = open('dialogue_alibaba2.txt', encoding='utf-8', mode='r')
    dialogue = dialogue.readlines()
    dialogue = [p for p in dialogue if p.startswith('1')]
    print(len(dialogue))
    pairs = get_pair(max_pair, dialogue)
    # break
    # print(pairs)
    data = []
    for p in pairs:
        data.append([p[0], p[1], 1])
    for i, p in enumerate(pairs):
        data.append([p[0], pairs[(i + 8) % len(pairs)][1], 0])
    df = pd.DataFrame(data, columns=['sentence_q', 'sentence_a', 'label'])

    print(len(data))
    return df



from parameter import  MAX_SENTENCE_LENGTH,MAX_FEATURES,embedding_size,lr,MAX_SENTENCE_LENGTH,n_hidden_units,batch_size

training_iters = 100000



n_classes = 2  # MNIST classes (0-9 digits)


def get_sentiment_data():
    df_sentiment = convert_dialogue_to_pair()
    df_sentiment.to_csv('alibaba.csv',index=False,columns=['label','sentence_q','sentence_a'],encoding='utf-8')
    print('=========finish convert ========')
    df_sentiment = df_sentiment.sample(frac=0.9,random_state=20)
    # df_sentiment = pd.read_csv('sentiment.csv', encoding='utf-8')
    # df_sentiment['sentence_q'] = df_sentiment['sentence']
    # df_sentiment['sentence_a'] = df_sentiment['sentence']
    sentenses_q = df_sentiment['sentence_q'].values
    sentenses_a = df_sentiment['sentence_a'].values
    sentenses = [s.lower() for s in sentenses_q + sentenses_a]
    wordlist_sentence = [nltk.word_tokenize(s) for s in sentenses]
    ws = []
    for wordlist in wordlist_sentence:
        ws.extend(wordlist)
    word_counter = Counter(ws)
    mc = word_counter.most_common(100)
    # print(mc)
    vocab_size = min(MAX_FEATURES, len(word_counter)) + 2
    word2index = {x[0]: i + 2 for i, x in
                  enumerate(word_counter.most_common(MAX_FEATURES))}
    word2index["PAD"] = 0
    word2index["UNK"] = 1
    index2word = {v: k for k, v in word2index.items()}
    res = []
    print('=========finish index word ========')
    print('iterrows')
    for line in df_sentiment.iterrows():
        # print('line')
        label, sentence = str(line[1]['label']), line[1]['sentence_q']
        # label, sentence = line.strip().split("\t")
        # print(label,sentence)
        # words = nltk.word_tokenize(sentence_q.lower())

        # words = nltk.word_tokenize(sentence.lower())
        words = sentence.split(' ')
        # print(words)
        seqs1 = []
        for word in words:
            if word in word2index.keys():
                seqs1.append(word2index[word])
            else:
                seqs1.append(word2index["UNK"])
        if MAX_SENTENCE_LENGTH < len(seqs1):
            # print('unexpected length of padding', len(padding))
            continue
        padding = [0] * (MAX_SENTENCE_LENGTH - len(seqs1))
        padding.extend(seqs1)
        # if len(padding) != MAX_SENTENCE_LENGTH:
        #     print('unexpected length of padding', len(padding))

        question = padding
        label, sentence = str(line[1]['label']), line[1]['sentence_a']
        # label, sentence = line.strip().split("\t")
        # print(label,sentence)
        # words = nltk.word_tokenize(sentence_q.lower())

        # words = nltk.word_tokenize(sentence.lower())
        words = sentence.split(' ')
        # print(words)
        seqs1 = []
        for word in words:
            if word in word2index.keys():
                seqs1.append(word2index[word])
            else:
                seqs1.append(word2index["UNK"])
        if MAX_SENTENCE_LENGTH < len(seqs1):
            # print('unexpected length of padding', len(padding))
            continue
        padding = [0] * (MAX_SENTENCE_LENGTH - len(seqs1))
        padding.extend(seqs1)
        # if len(padding) != MAX_SENTENCE_LENGTH:
        #     print('unexpected length of padding', len(padding))
        # padding = [u for u in padding]
        # for i in range(MAX_SENTENCE_LENGTH):

        answer = padding
        if label == '0':
            res.append([[1,0], question, answer])
            # print('0')
        if label == '1':
            res.append([[0,1], question, answer])
            # print('1')
    return res,vocab_size

data,vocab_size = get_sentiment_data()

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# x ==> x_q,x_a
x_q = tf.placeholder(tf.int32, [None, MAX_SENTENCE_LENGTH])
x_a = tf.placeholder(tf.int32, [None, MAX_SENTENCE_LENGTH])

W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
embedded_chars_q = tf.nn.embedding_lookup(W, x_q)
embedded_chars_a = tf.nn.embedding_lookup(W, x_a)
# embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)


y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # (28, 128)
    #'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    #'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X1, X2, name, weights, biases):
    # hidden layer for input to cell
    ########################################
    # tf.reset_default_graph()
    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    # X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    # X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    # X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    ##########################################

    # basic LSTM Cell.

    with tf.variable_scope('RNN' + name):
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            print('<12')
            cell = tf.nn.rnn_cell.LSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        else:
            cell = tf.contrib.rnn.LSTMCell(n_hidden_units)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    # You have 2 options for following step.
    # 1: tf.nn.rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # If use option 1, you have to modified the shape of X_in, go and check out this:
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
    # In here, we go for option 2.
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.
    outputs, final_state = tf.nn.dynamic_rnn(cell, X1, initial_state=init_state, time_major=False)
    # hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']
    # # or
    # unpack to list [(batch, outputs)..] * steps
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs1 = tf.unpack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs
    else:
        outputs1 = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    outputs, final_state = tf.nn.dynamic_rnn(cell, X2, initial_state=init_state, time_major=False)
    # hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']
    # # or
    # unpack to list [(batch, outputs)..] * steps
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs2 = tf.unpack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs
    else:
        outputs2 = tf.unstack(tf.transpose(outputs, [1, 0, 2]))

    return outputs1[-1], outputs2[-1]
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # shape = (128, 10)
    # return results


def generate_number_classification():
    import numpy as np
    import random
    number = training_iters
    data = []
    for i in range(number):
        number_list = []
        for j in range(MAX_SENTENCE_LENGTH):
            number_list.append(random.randint(0, MAX_FEATURES))
        # number_list.sort()
        # number_list = [str(n) for n in number_list]
        data.append(number_list)
    res = []
    for i, number in enumerate(data):
        if i % 2 == 0:
            question = [str(n) for n in number]
            res.append([[1, 0], question])
        if i % 2 == 1:
            question = [str(n + 30) for n in number]
            res.append([[0, 1], question])
    # training_data = pd.DataFrame(res, columns=['label', 'sentence_q', 'sentence_a'])
    return res
from sklearn.model_selection import train_test_split



input_x = [[ row[1], row[2]]for row in data]
Ys = [row[0]for row in data]
Xtrain, Xtest, ytrain, ytest = train_test_split(input_x,Ys,test_size = 0.1,random_state = 42,stratify = Ys)

def get_q_a(X):
    """ Return the train_test_split"""
    Xq = [row[0] for row in X]
    Xa = [row[1] for row in X]
    #y  = [row[0] for row in data]
    return Xq,Xa

training_iters = len(data)
print('{} pairs of dialogue'.format(training_iters))
data_copy = data[:]
for row in data:
    label,q,a = row
    if len(label)!=2 or len(q)!=MAX_SENTENCE_LENGTH  or len(a)!= MAX_SENTENCE_LENGTH:
        data_copy.remove(row)
        print('one invalid')
RNN_state_q, RNN_state_a = RNN(embedded_chars_q, embedded_chars_a, 'q', weights, biases)

# match_W  = tf.Variable(tf.random_normal([n_hidden_units, n_hidden_units]))
# match_b = tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ]))
# RNN_state_q = tf.matmul(RNN_state_q, match_W) + match_b
# product = tf.reduce_sum(tf.multiply(RNN_state_q, RNN_state_a),keepdims = True)
#product = tf.multiply(RNN_state_q, RNN_state_a)
difference = tf.subtract(RNN_state_q, RNN_state_a)
difference = tf.abs(difference)
two_class_W = tf.Variable(tf.random_normal([n_hidden_units, 2]))
two_class_b = tf.Variable(tf.constant(0.1, shape=[2, ]))
pred = tf.matmul(difference, two_class_W) + two_class_b

# concat is of no use
# concat_layer=tf.concat([RNN_state_q,RNN_state_a],axis=1)
# concat_W = tf.Variable(tf.random_normal([n_hidden_units*2, 2]))
# concat_b = tf.Variable(tf.constant(0.1, shape = [2, ]))
# pred = tf.matmul(concat_layer, concat_W) + concat_b


# dot product mapping
# two_class_W = tf.Variable(tf.random_normal([1, 2]))
# two_class_b = tf.Variable(tf.constant(0.1, shape = [2, ]))
# pred = tf.matmul(product, two_class_W)+ two_class_b

# pred = tf.matmul(RNN_state, weights['out']) + biases['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def get_validation(data,batch_size):
    data = data[-batch_size:]
    return [u[1] for u in data], [u[2] for u in data], [u[0] for u in data]

def get_batch(data_, step, batch_size):
    data = data_[step * batch_size:(step + 1) * batch_size]
    return [u[1] for u in data], [u[2] for u in data], [u[0] for u in data]

start = time.time()
with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while ((step + 2) * batch_size) < training_iters:
        # print('{},{},{},{}'.format(step,batch_size,training_iters,(step+1) * batch_size ))
        batch_xs_q, batch_xs_a, batch_ys = get_batch(data, step, batch_size)
        batch_xs_q = np.array(batch_xs_q)
        batch_xs_a = np.array(batch_xs_a)
        batch_ys= np.array(batch_ys)
        #batch_xs2_q, batch_xs2_a, batch_ys2 = get_validation(data,batch_size)#get_batch(data, step + 1, batch_size)
        batch_xs2_q, batch_xs2_a, = get_q_a(Xtest[-batch_size:])
        batch_ys2 = ytest[-batch_size:]

        batch_xs2_q = np.array(batch_xs2_q)
        batch_xs2_a = np.array(batch_xs2_a)
        batch_ys2= np.array(batch_ys2)

        # mnist.train.next_batch(batch_size)
        # batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        # batch_xs = batch_xs.reshape([batch_size, n_steps])
        sess.run([train_op], feed_dict={
            x_q: batch_xs_q,
            x_a: batch_xs_a,
            y: batch_ys,
        })
        if step % 20 == 0:
            print('\n--------TRAIN--------:  No.', (step) * batch_size)
            print((step) * batch_size, sess.run(accuracy, feed_dict={
                x_q: batch_xs2_q,
                x_a: batch_xs2_a,
                y: batch_ys2,
            }))
        if step >10000:
            break
        step += 1
end = time.time()
print('finish  training  ,use {} sec'.format(int(end - start)))