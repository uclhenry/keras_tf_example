import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import nltk
import pandas as pd
from collections import Counter
import numpy as np
MAX_FEATURES = 150
MAX_SENTENCE_LENGTH =100

# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 127
vocab_size = 200
embedding_size = 100
n_inputs = embedding_size   # MNIST data input (img shape: 28*28)
n_steps = MAX_SENTENCE_LENGTH    # time steps
n_hidden_units = 128  # neurons in hidden layer
n_classes = 2     # MNIST classes (0-9 digits)

def get_sentiment_data():
    df_sentiment = pd.read_csv('sentiment.csv',encoding='utf-8')
    sentenses = df_sentiment['sentence'].values
    sentenses = [s.lower() for s in sentenses]
    wordlist_sentence = [nltk.word_tokenize(s) for s in sentenses]
    ws = []
    for wordlist in wordlist_sentence:
        ws.extend(wordlist)
    word_counter = Counter(ws)
    mc = word_counter.most_common(100)
    print(mc)
    vocab_size = min(MAX_FEATURES, len(word_counter)) + 2
    word2index = {x[0]: i + 2 for i, x in
                  enumerate(word_counter.most_common(MAX_FEATURES))}
    word2index["PAD"] = 0
    word2index["UNK"] = 1
    index2word = {v: k for k, v in word2index.items()}
    res = []
    print('iterrows')
    for line in df_sentiment.iterrows():
        #print('line')
        label, sentence = str(line[1]['label']), line[1]['sentence']
        # label, sentence = line.strip().split("\t")
        # print(label,sentence)
        # words = nltk.word_tokenize(sentence_q.lower())

        words = nltk.word_tokenize(sentence.lower())
        #print(words)
        seqs1 = []
        for word in words:
            if word in word2index.keys():
                seqs1.append(word2index[word])
            else:
                seqs1.append(word2index["UNK"])
        if MAX_SENTENCE_LENGTH < len(seqs1):
            print('unexpected length of padding', len(padding), padding)
            continue
        padding = [0]*(MAX_SENTENCE_LENGTH - len(seqs1))
        padding.extend(seqs1)
        if len(padding)!=MAX_SENTENCE_LENGTH:
            print('unexpected length of padding',len(padding),padding)
        #padding = [u for u in padding]
        #for i in range(MAX_SENTENCE_LENGTH):
        if label == '0':
            res.append([np.array([1, 0]), padding])
            #print('0')
        if label == '1':
            res.append([np.array([0, 1]), padding])
            #print('1')
    return res

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# tf Graph input
x = tf.placeholder(tf.int32, [None, n_steps])

W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name = "W")
embedded_chars = tf.nn.embedding_lookup(W, x)
#embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)


y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    #X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    #X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    #X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    ##########################################

    # basic LSTM Cell.
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
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
    outputs, final_state = tf.nn.dynamic_rnn(cell, X, initial_state=init_state, time_major=False)
    # hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']
    # # or
    # unpack to list [(batch, outputs)..] * steps
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)
    return results

def generate_number_classification():
    import numpy as np
    import random
    number = training_iters
    data = []
    for i in range(number):
        number_list = []
        for j in range(MAX_SENTENCE_LENGTH):
            number_list.append(random.randint(0,MAX_FEATURES))
        #number_list.sort()
        #number_list = [str(n) for n in number_list]
        data.append(number_list)
    res = []
    for i,number in enumerate(data):
        if i %2 ==0:
            question = [str(n) for n in number]
            res.append([[1,0],question])
        if i %2 ==1:
            question = [str(n+30) for n in number]
            res.append([[0,1], question])
    #training_data = pd.DataFrame(res, columns=['label', 'sentence_q', 'sentence_a'])
    return res

data = get_sentiment_data()
training_iters = len(data)
pred = RNN(embedded_chars, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def get_batch(data,step , batch_size):
    data = data[step*batch_size:(step+1)*batch_size]
    return [u[1] for u in data],[u[0] for u in data]

with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while ((step+2) * batch_size )< training_iters:
        #print('{},{},{},{}'.format(step,batch_size,training_iters,(step+1) * batch_size ))
        batch_xs, batch_ys = get_batch(data,step , batch_size)
        batch_xs2, batch_ys2 = get_batch(data,step +1, batch_size)
        # mnist.train.next_batch(batch_size)
        #batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        #batch_xs = batch_xs.reshape([batch_size, n_steps])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 2 == 0:
            print((step) * batch_size,sess.run(accuracy, feed_dict={
            x: batch_xs2,
            y: batch_ys2,
            }))
        step += 1
