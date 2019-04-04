
import tensorflow as tf
import numpy as np


class TextCNN:
    def __init__(self, filter_sizes,num_filters,num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,vocab_size,embed_size,
                 is_training,initializer=tf.random_normal_initializer(stddev=0.1),multi_label_flag=False,clip_gradients=5.0,decay_rate_big=0.50):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length=sequence_length
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.is_training=is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")#ADD learning_rate
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * decay_rate_big)
        self.filter_sizes=filter_sizes # it is a list of int. e.g. [3,4,5]
        self.num_filters=num_filters
        self.initializer=initializer
        self.num_filters_total=self.num_filters * len(filter_sizes) #how many filters totally.
        self.multi_label_flag=multi_label_flag
        self.clip_gradients = clip_gradients

        # add placeholder (X,label)
        self.input_x_q = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        self.input_x_a = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        self.input_y = tf.placeholder(tf.int32, [None,],name="input_y")  # y:[None,num_classes]
        self.input_y_multilabel = tf.placeholder(tf.float32,[None,self.num_classes], name="input_y_multilabel")  # y:[None,num_classes]. this is for multi-label classification only.
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")
        self.epoch_increment=tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        self.logits = self.inference() #[None, self.label_size]. main computation graph is here.
        if not is_training:
            return
        if multi_label_flag:
            print("going to use multi label loss.")
            self.loss_val = self.loss_multilabel()
        else:
            print("going to use single label loss.")
            self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, 1, name="predictions")  # shape:[None,]

        if not self.multi_label_flag:
            correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32), self.input_y) #tf.argmax(self.logits, 1)-->[batch_size]
            self.accuracy =tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy") # shape=()
        else:
            self.accuracy = tf.constant(0.5) #fuke accuracy. (you can calcuate accuracy outside of graph using method calculate_accuracy(...) in train.py)

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"): # embedding matrix
            self.Embedding_q = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size], initializer=self.initializer) #[vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            #self.Embedding_a = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size], initializer=self.initializer) #[vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.W_projection = tf.get_variable("W_projection",shape=[self.num_filters_total, self.num_classes],initializer=self.initializer) #[embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])       #[label_size] #ADD 2017.06.09

    def inference(self):
        """main computation graph here: 1.embedding-->2.average-->3.linear classifier"""
        # 1.=====>get emebedding of words in the sentence
        self.embedded_words_q = tf.nn.embedding_lookup(self.Embedding_q, self.input_x_q)#[None,sentence_length,embed_size]
        self.embedded_words_a = tf.nn.embedding_lookup(self.Embedding_q, self.input_x_a)#[None,sentence_length,embed_size]
        self.sentence_expanded_q=tf.expand_dims(self.embedded_words_q, -1) #[None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
        self.sentence_expanded_a=tf.expand_dims(self.embedded_words_a, -1) #[None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
        # ？ 30，100,1
        # 2.=====>loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
        pooled_outputs_q = []
        pooled_outputs_a = []
        for i,filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution-pooling-%s" %filter_size):
                # ====>a.create filter
                # 3 100 ,1 , 128
                filter = tf.get_variable("filter-%s" % filter_size,[filter_size,self.embed_size,1,self.num_filters],initializer=self.initializer)
                # ====>b.conv operation: conv2d===>computes a 2-D convolution given 4-D `input` and `filter` tensors.
                #Conv.Input: given an input tensor of shape `[batch, in_height, in_width, in_channels]` and a filter / kernel tensor of shape `[filter_height, filter_width, in_channels, out_channels]`
                #Conv.Returns: A `Tensor`. Has the same type as `input`.
                #         A 4-D tensor. The dimension order is determined by the value of `data_format`, see below for details.
                #1)each filter with conv2d's output a shape:[1,sequence_length-filter_size+1,1,1];2)*num_filters--->[1,sequence_length-filter_size+1,1,num_filters];3)*batch_size--->[batch_size,sequence_length-filter_size+1,1,num_filters]
                #input data format:NHWC:[batch, height, width, channels];output:4-D
                #? ,28 ,1 , 128
                conv_q=tf.nn.conv2d(self.sentence_expanded_q, filter, strides=[1, 1, 1, 1], padding="VALID", name="conv") #shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                conv_a=tf.nn.conv2d(self.sentence_expanded_a, filter, strides=[1, 1, 1, 1], padding="VALID", name="conv") #shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                # ====>c. apply nolinearity
                #128 ,
                b=tf.get_variable("b-%s"%filter_size,[self.num_filters]) #ADD 2017-06-09
                h_q=tf.nn.relu(tf.nn.bias_add(conv_q,b),"relu") #shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                h_a=tf.nn.relu(tf.nn.bias_add(conv_a,b),"relu") #shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                # ====>. max-pooling.  value: A 4-D `Tensor` with shape `[batch, height, width, channels]
                #                  ksize: A list of ints that has length >= 4.  The size of the window for each dimension of the input tensor.
                #                  strides: A list of ints that has length >= 4.  The stride of the sliding window for each dimension of the input tensor.
                pooled_q = tf.nn.max_pool(h_q, ksize=[1,self.sequence_length-filter_size+1,1,1], strides=[1,1,1,1], padding='VALID',name="pool")#shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.
                pooled_a = tf.nn.max_pool(h_a, ksize=[1,self.sequence_length-filter_size+1,1,1], strides=[1,1,1,1], padding='VALID',name="pool")#shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.
                pooled_outputs_q.append(pooled_q)
                pooled_outputs_a.append(pooled_a)
        # 3.=====>combine all pooled features, and flatten the feature.output' shape is a [1,None]
        #e.g. >>> x1=tf.ones([3,3]);x2=tf.ones([3,3]);x=[x1,x2]
        #         x12_0=tf.concat(x,0)---->x12_0' shape:[6,3]
        #         x12_1=tf.concat(x,1)---->x12_1' shape;[3,6]
        self.h_pool_q=tf.concat(pooled_outputs_q, 3) #shape:[batch_size, 1, 1, num_filters_total]. tf.concat=>concatenates tensors along one dimension.where num_filters_total=num_filters_1+num_filters_2+num_filters_3
        self.h_pool_a=tf.concat(pooled_outputs_a, 3) #shape:[batch_size, 1, 1, num_filters_total]. tf.concat=>concatenates tensors along one dimension.where num_filters_total=num_filters_1+num_filters_2+num_filters_3
        self.h_pool_flat_q=tf.reshape(self.h_pool_q, [-1, self.num_filters_total]) #shape should be:[None,num_filters_total]. here this operation has some result as tf.sequeeze().e.g. x's shape:[3,3];tf.reshape(-1,x) & (3, 3)---->(1,9)
        self.h_pool_flat_a=tf.reshape(self.h_pool_a, [-1, self.num_filters_total]) #shape should be:[None,num_filters_total]. here this operation has some result as tf.sequeeze().e.g. x's shape:[3,3];tf.reshape(-1,x) & (3, 3)---->(1,9)

        #4.=====>add dropout: use tf.nn.dropout
        with tf.name_scope("dropout"):
            self.h_drop_q  =  tf.nn.dropout(self.h_pool_flat_q, keep_prob=self.dropout_keep_prob) #[None,num_filters_total]
            self.h_drop_a  =  tf.nn.dropout(self.h_pool_flat_a, keep_prob=self.dropout_keep_prob) #[None,num_filters_total]

        #5. logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            difference = tf.subtract(self.h_drop_q, self.h_drop_a)
            self.difference = tf.abs(difference)

            logits = tf.matmul(self.difference, self.W_projection) + self.b_projection  #shape:[None, self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
        return logits

    def loss(self,l2_lambda=0.0001):#0.001
        with tf.name_scope("loss"):
            #input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits);#sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            #print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
            loss=tf.reduce_mean(losses)#print("2.loss.loss:", loss) #shape=()
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses
        return loss

    def loss_multilabel(self,l2_lambda=0.00001): #0.0001#this loss function is for multi-label classification
        with tf.name_scope("loss"):
            #input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            #input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel, logits=self.logits);#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            #losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            print("sigmoid_cross_entropy_with_logits.losses:",losses) #shape=(?, 1999).
            losses=tf.reduce_sum(losses,axis=1) #shape=(?,). loss for all data in the batch
            loss=tf.reduce_mean(losses)         #shape=().   average loss in the batch
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op


import numpy as np
import pandas as pd
import nltk
from collections import Counter
from parameter import MAX_SENTENCE_LENGTH,MAX_FEATURES,embedding_size

# hyperparameters

training_iters = 100000



def get_sentiment_data():
    df_sentiment = pd.read_csv('sentiment.csv', encoding='utf-8')
    sentenses = df_sentiment['sentence'].values
    sentenses = [s.lower() for s in sentenses]
    wordlist_sentence = [nltk.word_tokenize(s) for s in sentenses]
    ws = []
    for wordlist in wordlist_sentence:
        ws.extend(wordlist)
    word_counter = Counter(ws)
    mc = word_counter.most_common(100)
    #print(mc)
    vocab_size = min(MAX_FEATURES, len(word_counter)) + 2
    word2index = {x[0]: i + 2 for i, x in
                  enumerate(word_counter.most_common(MAX_FEATURES))}
    word2index["PAD"] = 0
    word2index["UNK"] = 1
    index2word = {v: k for k, v in word2index.items()}
    res = []
    print('iterrows')
    for line in df_sentiment.iterrows():
        # print('line')
        label, sentence = str(line[1]['label']), line[1]['sentence']
        # label, sentence = line.strip().split("\t")
        # print(label,sentence)
        # words = nltk.word_tokenize(sentence_q.lower())

        words = nltk.word_tokenize(sentence.lower())
        # print(words)
        seqs1 = []
        for word in words:
            if word in word2index.keys():
                seqs1.append(word2index[word])
            else:
                seqs1.append(word2index["UNK"])
        if MAX_SENTENCE_LENGTH < len(seqs1):
            print('unexpected length of padding', len(padding), padding)
            continue
        padding = [0] * (MAX_SENTENCE_LENGTH - len(seqs1))
        padding.extend(seqs1)
        if len(padding) != MAX_SENTENCE_LENGTH:
            print('unexpected length of padding', len(padding), padding)
        # padding = [u for u in padding]
        # for i in range(MAX_SENTENCE_LENGTH):
        if label == '0':
            res.append([1, padding])
            # print('0')
        if label == '1':
            res.append([0, padding])
            # print('1')
    return res,vocab_size

from parameter import max_pair


#test started
def get_q_a(X):
    """ Return the train_test_split"""
    Xq = [row[0] for row in X]
    Xa = [row[1] for row in X]
    # y  = [row[0] for row in data]
    return Xq, Xa
def test():
    from parameter import  lr,batch_size
    from sklearn.model_selection import train_test_split
    #below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.
    num_classes=2
    learning_rate=lr

    decay_steps=1000
    decay_rate=0.9
    #sequence_length=MAX_SENTENCE_LENGTH

    is_training=True
    dropout_keep_prob=1 #0.5
    filter_sizes=[3,4]
    num_filters=128
    from alime_data import get_alibaba
    data,vocab_size = get_alibaba(max_pair)
    X = [[q,a]for l,q,a in data]
    Ys = [l for l,q,a in data]
    from sklearn.model_selection import train_test_split
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, Ys, test_size=0.1, random_state=42, stratify=Ys)
    import time
    start = time.time()
    textRNN = TextCNN(filter_sizes,num_filters,num_classes, learning_rate, batch_size, decay_steps, decay_rate,MAX_SENTENCE_LENGTH,vocab_size,embedding_size,is_training)
    merged_summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("logs/", sess.graph)
        for step in range(1000000):
            if (step+1)*batch_size >len(Xtrain):
                print('test No.',(step+1)*batch_size )
                break
            input_x_a = [a for q, a in Xtrain[step*batch_size:(step+1)*batch_size]]
            input_x_q = [q for q, a in Xtrain[step*batch_size:(step+1)*batch_size]]
            input_y = ytrain[step*batch_size:(step+1)*batch_size]
            input_x_a2 = [a for q, a in Xtest[-batch_size:]]
            input_x_q2 = [q for q, a in Xtest[-batch_size:]]
            input_y2 = ytest[-batch_size:]

            loss,acc,_,summary=sess.run(
                [textRNN.loss_val,textRNN.accuracy,textRNN.train_op,merged_summary_op],
                feed_dict={
                    textRNN.input_x_q:input_x_q,
                    textRNN.input_x_a:input_x_a,
                    textRNN.input_y:input_y,
                    textRNN.dropout_keep_prob: dropout_keep_prob
                })
            tf.summary.scalar('training_accuracy',acc)
            if step% 20!=0:continue
            print('\n--------TRAIN--------:  No.', (step) * batch_size)
            #print("loss:",loss,"acc:",acc)#,"label:",input_y,"prediction:",predict
            print((step) * batch_size, sess.run(textRNN.accuracy, feed_dict={
                textRNN.input_x_q: input_x_q2,
                textRNN.input_x_a: input_x_a2,
                textRNN.input_y: input_y2,
                textRNN.dropout_keep_prob: dropout_keep_prob
            }))
        writer.close()
    end = time.time()
    print('finish  training  ,use {} sec'.format(int(end - start)))
            #print("W_projection_value_:",W_projection_value)
test()