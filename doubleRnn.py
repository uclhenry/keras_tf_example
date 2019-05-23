from keras.layers.wrappers import Bidirectional
from keras.models import Model
import keras.backend as K
from keras.layers import Dense,Input
from keras.layers.core import Dense,Lambda
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
import numpy as np
from keras.layers import Flatten,Reshape,Permute

def reduce_mean(X):
    return K.mean(X,axis = -1)
rnn_hidden_size = 100
dropout_rate = 0.1
rs = True
MAX_SENTENCE_LENGTH = 10
turn_number = 5
data_size  = 1000
embedding_size = 50
X = np.random.random((data_size,turn_number*MAX_SENTENCE_LENGTH,embedding_size))
Y = []
n_class = 5
for i in range(data_size):
    r  =[0]*5
    r[np.random.randint(0,n_class)] = 1
    Y.append(r)
Y = np.array(Y)
x_utterance_concat = Input(shape =(turn_number * MAX_SENTENCE_LENGTH,embedding_size  ), name="concat_utter")
rnn1 = Bidirectional(
            LSTM(rnn_hidden_size, dropout=dropout_rate, recurrent_dropout=dropout_rate, name='RNN',
                 return_sequences=rs))(x_utterance_concat)
rnn1_shape = Reshape((turn_number,MAX_SENTENCE_LENGTH,rnn_hidden_size*2) )(rnn1)
swap = Permute((1,3,2))(rnn1_shape)
average = Lambda(reduce_mean)(swap)
rnn2 = Bidirectional(
            LSTM(rnn_hidden_size, dropout=dropout_rate, recurrent_dropout=dropout_rate, name='RNN',
                 return_sequences=False))(average)
class_layer = Dense(n_class, activation='softmax')(rnn2)
model = Model(inputs=x_utterance_concat, outputs=class_layer)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, batch_size=30,
               epochs=10,
               validation_data=(X[-50:], Y[-50:]))