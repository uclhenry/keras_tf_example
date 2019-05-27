from keras.layers.wrappers import Bidirectional
from keras.models import Model

import tensorflow as tf
import keras.backend as K
from keras.models import load_model
from keras.layers import Dense, Input
from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Flatten, Reshape, Permute

from keras.layers.core import Dense, Lambda
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import numpy as np
from load_CamRest import get_X_Y_from_raw_text, slot_state_dict, dict_to_feature
from src.config.dst_worker_config import prefix_request, prefix_info

informable = ['area', 'food', 'pricerange']
requestable = informable + ['phone', 'postcode', 'address']
import random
import collections
import nltk
import os


class ActionModel():
    def __init__(self, parameter=None):
        self.word_embedding_path = 'D://udc//data//glove.6B.100d.txt'
        if parameter == None:
            self.default_config()
            self.auto_set_nclass()
            return
        self.model = None
        self.hidden_size = parameter['hidden_size']
        self.input_shape = parameter['input_shape']
        self.batch_size = parameter['batch_size']
        self.num_epochs = parameter['num_epochs']
        self.all_action = parameter['actions']
        self.loss_d = {k: 'binary_crossentropy' for k in self.all_action}
        self.lossweight_d = {k: 1. for k in self.all_action}
        self.set_slot()


    def default_config(self):
        self.is_binary_model = False
        self.max_vocab_size = 1200
        self.verbose = 0
        self.model = None
        self.hidden_size = 300
        self.input_shape = (9,)
        self.batch_size = 20
        self.num_epochs = 6
        self.X_feature_number = 9
        self.model_path = "D://model/policy_learner_single.h5"
        more_actions = ['action_goodbye', 'action_morehelp', 'action_inform_address', \
                        'action_inform_food', 'action_inform_phone', \
                        'action_inform_area', 'action_inform_postcode', 'action_search_rest']
        self.all_action = ['action_ask_area', 'action_ask_pricerange', 'action_ask_food'] + more_actions
        self.loss_d = {k: 'binary_crossentropy' for k in self.all_action}
        self.lossweight_d = {k: 1. for k in self.all_action}
        self.set_slot()
        self.allow_action = self.all_action

    def auto_set_nclass(self):
        if self.is_binary_model:
            self.n_class = 2
        else:
            self.n_class = len(self.all_action)
    # keras
    def set_verbose(self, verbose):
        self.verbose = verbose

    def concat(self, X):
        return K.concatenate(X, axis=-1)

    def reduce_mean(self, X):
        return K.mean(X, axis=-1)

    def fix_length_vector(self, v, n, forward=True):
        if len(v) > n:
            if forward:
                return v[:n]
            else:
                return v[-n:]
        else:
            return [0] * (n - len(v)) + v

    def words2seq(self, words, word2index):
        seqs = []
        for word in words:
            if word in word2index.keys():
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        return seqs

    def get_action_label_dict(self, Y):
        # Y is list
        y_dict = dict()
        for action in self.all_action:
            y_dict[action] = []
            for labels in Y:
                if action in labels:
                    y_dict[action].append(1)
                else:
                    y_dict[action].append(0)
        return y_dict

    def set_word_index_from_text(self, last_utterances):
        word_freqs = collections.Counter()
        maxlen = 0
        for last_utterance in last_utterances:
            words = nltk.word_tokenize(last_utterance.lower())
            if len(words) > maxlen:
                maxlen = len(words)
            for word in words:
                word_freqs[word] += 1
        vocab_size = min(self.max_vocab_size, len(word_freqs)) + 2
        print('vocab size is {}'.format(vocab_size))
        word2index = {x[0]: i + 2 for i, x in enumerate(word_freqs.most_common(self.max_vocab_size))}
        word2index["PAD"] = 0
        word2index["UNK"] = 1
        self.word2index = word2index
        self.index2word = {v: k for k, v in word2index.items()}

    def build_counter(self, history):
        last_utterances = [his[-1] for his in history]
        self.set_word_index_from_text(last_utterances)

    # glove字典，每个单词对应一个100维的向量
    def get_glove_dict(self):
        embeddings_index = {}
        # f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
        f = open(self.word_embedding_path, encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Found %s word vectors.' % len(embeddings_index))
        return embeddings_index

    def get_embed_matrix(self, word_index, EMBEDDING_DIM):
        GLOVE_DIR = 'text_data/glove.6B'
        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        embedding_index = self.get_glove_dict()
        for word, i in word_index.items():
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def set_epoch(self, n):
        self.num_epoch = n

    def set_slot(self):
        self.informable = informable
        self.requestable = requestable

    def load_model(self, path):
        print("Using loaded model to predict...")
        m = load_model(path)
        self.model = m
        return m

    def single_label(self, input_x, name):
        layer1 = Dense(self.hidden_size)(input_x)
        layer2 = Dense(self.hidden_size)(layer1)
        layer2 = Dense(self.hidden_size)(layer2)
        binary1 = Dense(1, activation="sigmoid", name=name)(layer2)
        return binary1

    def build_model(self):
        label_layers = dict()
        x = Input(shape=self.input_shape, name="input")
        for name in self.all_action:
            label_layers[name] = self.single_label(x, name)
        print("Training...")
        output_layers_list = [label_layers[u] for u in self.all_action]
        model = Model(inputs=x, outputs=output_layers_list)
        model.compile(optimizer="adam",
                      loss=self.loss_d,
                      loss_weights=self.lossweight_d,
                      metrics=['accuracy'])
        self.model = model

    def get_multi_labels(self, Y):
        labels_y = [[y[column_id] for y in Y] for column_id in range(len(self.all_action))]
        return labels_y

    def train_single_action(self, X, Y):
        # print("Training... train_single_action")
        # print('{} labels for actions :{}'.format(len(self.all_action),' '.join(self.all_action)))
        # print('labels in data is ',len(Y[0]))
        X = np.array(X)
        # Split input into training and test
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2,
                                                        random_state=42)
        self.model.fit(Xtrain, ytrain, batch_size=self.batch_size, verbose=self.verbose,
                       epochs=self.num_epochs,
                       validation_data=(Xtest, ytest))
        return Xtrain, Xtest, ytrain, ytest

    def train(self, X, Y):
        print("Training...")
        X = np.array(X)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.1,
                                                        random_state=42)
        ytrain = self.get_multi_labels(ytrain)
        ytest = self.get_multi_labels(ytest)
        self.model.fit(Xtrain, ytrain, batch_size=self.batch_size, verbose=self.verbose,
                       epochs=self.num_epochs,
                       validation_data=(Xtest, ytest))

    def get_feature_from_nlu(self, slu, informable, requestable):
        states = slot_state_dict(informable, requestable)
        if slu != None:
            for user_act in slu:
                if user_act['act'] == 'request':
                    slot_name = user_act['slots'][0][1]
                    if 'request_' + slot_name in states.keys():
                        states['request_' + slot_name] = 1
                if user_act['act'] == 'inform':
                    slot_name = user_act['slots'][0][0]
                    if 'inform_' + slot_name in states.keys():
                        states['inform_' + slot_name] = 1
        # prefix_info = ['inform_'+ u for u in informable]
        # prefix_request = ['request_'+ u for u in requestable]
        state_feature = dict_to_feature(states, prefix_request + prefix_info)
        return state_feature

    def save(self):
        self.model.save(self.model_path)

    def predict(self, slu_dict):
        x = self.get_feature_from_nlu(slu_dict, self.informable, self.requestable)
        x = np.array(x)
        prediction = self.model.predict_on_batch(x.reshape(1, self.X_feature_number))
        print(len(prediction), len(prediction[0]))
        print(len(self.all_action))
        multi_action_prop_dict = dict()
        for i, action in enumerate(self.all_action):
            prop = prediction[i][0][0]
            print(action, prop)
            multi_action_prop_dict[action] = float(prop)
        return multi_action_prop_dict

    def load_data_and_train(self):
        X, Y, data, state_his_s, labels = get_X_Y_from_raw_text()
        data = [d for d in data if d[1] != []]
        Y = [d[1] for d in data]
        X = [d[0] for d in data]
        y_dict = dict()
        for action in self.all_action:
            y_dict[action] = []
            for labels in Y:
                if action in labels:
                    y_dict[action].append(1)
                else:
                    y_dict[action].append(0)

        self.build_binary_model()
        X_, Y_ = self.downsample(X, y_dict['action_inform_postcode'])
        self.train_single_action(X_, Y_)
        self.save()

    def downsample(self, X, Y):
        # get 1 and 0 y
        import numpy as np
        posi, nega = [], []
        for x, y in zip(X, Y):
            if y == 0:
                nega.append([x, y])
            else:
                posi.append([x, y])
        low_n = min(len(posi), len(nega))
        np.random.seed(1000)
        posi_order = np.random.permutation(len(posi))
        nega_order = np.random.permutation(len(nega))
        new_data = []
        t = 0
        for id in posi_order:
            new_data.append(posi[id])
            t += 1
            if t == low_n:
                break
        t = 0
        for id in nega_order:
            new_data.append(nega[id])
            t += 1
            if t == low_n:
                break
        return [d[0] for d in new_data], [d[1] for d in new_data]

    def single_action_prop_from_multi(self, action_prop):
        import math
        actions = [u for u in action_prop.keys()]
        props = [action_prop[action] for action in actions]
        props_exp = [math.exp(i) for i in props]
        sum_props_exp = sum(props_exp)
        softmax = [round(exp / sum_props_exp, 3) for exp in props_exp]
        return {action: prop for action, prop in zip(actions, softmax)}

    def get_best_action(self, action_prop_dict):
        best_action = None
        max_p = 0
        for action, p in action_prop_dict.items():
            if action not in self.allow_action:
                continue
            if p > max_p:
                best_action = action
                max_p = p
        return best_action

    def get_top_k_action(self, slu_dict):
        action_prop_dict = self.predict(slu_dict)
        action_prop_dict = sorted(action_prop_dict.items(), key=lambda d: d[1], reverse=True)
        return action_prop_dict

    def get_feature_from_slots(self, slots, informable, requestable):
        states = slot_state_dict(informable, requestable)
        if slots != None:
            for slot in slots:
                states[slot.type_name] = 1
        state_feature = dict_to_feature(states, prefix_request + prefix_info)
        print(state_feature)
        return state_feature

    def get_next_action_from_slots(self, slots):
        x = self.get_feature_from_slots(slots, self.informable, self.requestable)
        x = np.array(x)
        prediction = self.model.predict_on_batch(x.reshape(1, self.X_feature_number))
        multi_action_prop_dict = dict()
        for i, action in enumerate(self.all_action):
            prop = prediction[i][0][0]
            # print(action,prop)
            multi_action_prop_dict[action] = float(prop)
        return self.get_best_action(multi_action_prop_dict)

    def get_next_action(self, slu_dict):
        props = self.predict(slu_dict)
        return self.get_best_action(props)


# multiple model for different catogeory
class BinaryModel(ActionModel):

    def load_models(self):
        models = dict()
        for action in self.all_action:
            model = self.load_model(self.action_model_path(action))
            models[action] = model
        self.models = models

    def save(self, model_path):
        self.model.save(model_path)

    def get_model_path(self):
        return self.model_path

    def action_model_path(self,action,path=None):
        if not path:
            path = self.get_model_path()
        dir = path.split('.')[0]
        return dir + action + '.' + path.split('.')[-1]

    def build_binary_model(self):
        x = Input(shape=self.input_shape, name="input")
        layer1 = Dense(self.hidden_size)(x)
        layer2 = Dense(self.hidden_size)(layer1)
        layer3 = Dense(self.hidden_size)(layer2)
        binary = Dense(1, activation='sigmoid')(layer3)
        model = Model(inputs=x, outputs=binary)
        model.compile(optimizer="adam",
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        self.model = model
        return binary

    def show_report(self, stat_d):
        for action in self.all_action:
            print('report for action {}'.format(action))
            print(stat_d.get(action))

    def binary_train(self, X, y_dict):
        stat_d = dict()
        for action in self.all_action:
            print('training ', action)
            X_, Y_ = self.downsample(X, y_dict[action])
            Xtrain, Xtest, ytrain, ytest = self.train_single_action(X_, Y_)
            y_hat = self.model.predict_on_batch(Xtest)
            y_hat = [y[0] for y in y_hat.tolist()]
            to_int = lambda x: 1 if x > 0.5 else 0
            y_hat = [to_int(y) for y in y_hat]
            stat = classification_report(ytest, y_hat)
            stat_d[action] = stat
            dir = self.model_path.split('.')[0]
            path = dir + action + '.' + self.model_path.split('.')[-1]
            self.save(path)
        return stat_d

    def binary_tuning(self, X, y_dict):
        stat_d = dict()
        for action in self.all_action:
            print('training ', action)
            X_, Y_ = self.downsample(X, y_dict[action])
            Xtrain, Xtest, ytrain, ytest = self.train_single_action(X_, Y_)
            y_hat = self.model.predict_on_batch(Xtest)
            y_hat = [y[0] for y in y_hat.tolist()]
            to_int = lambda x: 1 if x > 0.5 else 0
            y_hat = [to_int(y) for y in y_hat]
            f1 = f1_score(ytest, y_hat)
            stat_d[action] = f1

        return stat_d

    def load_data_and_train(self):
        X, Y, data, state_his_s, labels, uss = get_X_Y_from_raw_text()
        data = [d for d in data if d[1] != []]
        Y = [d[1] for d in data]
        X = [d[0] for d in data]
        y_dict = self.get_action_label_dict()
        self.build_binary_model()
        self.binary_train(X, y_dict)
        self.show_report(self.stat_d)

    def get_next_action_from_slots(self, slots):
        x = self.get_feature_from_slots(slots, self.informable, self.requestable)
        x = np.array(x)
        multi_action_prop_dict = dict()
        for i, action in enumerate(self.all_action):
            model = self.models[action]
            prediction = model.predict_on_batch(x.reshape(1, self.X_feature_number))
            prop = prediction[0][0]
            multi_action_prop_dict[action] = float(prop)
        return self.get_best_action(multi_action_prop_dict)


class RnnBinaryModel(BinaryModel):

    def default_config(self):
        self.x_feature = 9
        self.is_binary_model = True
        self.model = None
        self.hidden_size = 100
        self.rnn_hidden_size = 1

        self.config()
        self.X_feature_number = 9
        self.input_shape = (self.turn_num, self.X_feature_number)
        self.batch_size = 100
        self.num_epochs = 6

        self.model_path = "D://model/policy_learner_single.h5"
        more_actions = ['action_goodbye', 'action_morehelp', 'action_inform_address', \
                        'action_inform_food', 'action_inform_phone', \
                        'action_inform_area', 'action_inform_postcode', 'action_search_rest']
        self.all_action = ['action_ask_area', 'action_ask_pricerange', 'action_ask_food'] + more_actions
        self.loss_d = {k: 'binary_crossentropy' for k in self.all_action}
        self.lossweight_d = {k: 1. for k in self.all_action}
        self.set_slot()
        self.allow_action = self.all_action
        self.hidden_size_range = range(10, 150, 15)
        self.rnn_size_range = range(10, 150, 15)

    def history_padding(self, state_his_s, feature_number,forward = False):
        new_data = []
        for state_his in state_his_s:
            if len(state_his) < self.turn_num:
                d = self.turn_num - len(state_his)
                left_pad = [[0] * feature_number for i in range(d)]
                left_pad.extend(state_his)
                new_data.append(left_pad)
            elif len(state_his) == self.turn_num:
                new_data.append(state_his)
            else:
                if not forward:
                    new_data.append(state_his[-self.turn_num:])
                else:
                    new_data.append(state_his[:self.turn_num])
        return new_data

    def config(self):
        self.turn_num = 3

    def build_binary_model(self):

        x = Input(shape=(self.turn_num, self.X_feature_number), name="input")
        rnn_hidden_size = 60
        dropout_rate = 0.2
        encoded_Q = Bidirectional(
            LSTM(self.rnn_hidden_size, dropout=dropout_rate, recurrent_dropout=dropout_rate, name='RNN',
                 return_sequences=False))(x)
        layer1 = Dense(self.hidden_size)(encoded_Q)
        layer2 = Dense(self.hidden_size)(layer1)
        layer3 = Dense(self.hidden_size)(layer2)
        binary = Dense(1, activation='sigmoid')(layer3)
        model = Model(inputs=x, outputs=binary)
        model.compile(optimizer="adam",
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        self.model = model
        return binary

    def load_data_and_train(self):
        X, Y, data, state_his_s, labels = get_X_Y_from_raw_text()
        new_data = []

        for state_his in state_his_s:
            if len(state_his) < self.turn_num:
                d = self.turn_num - len(state_his)
                left_pad = [[0] * self.X_feature_number for i in range(d)]
                left_pad.extend(state_his)
                new_data.append(left_pad)
            elif len(state_his) == self.turn_num:
                new_data.append(state_his)
            else:
                new_data.append(state_his[-self.turn_num:])

        data = [d for d in zip(new_data, labels) if d[1] != []]
        Y = [d[1] for d in data]
        X = [d[0] for d in data]
        y_dict = self.get_action_label_dict(Y)
        self.build_binary_model()
        stat_d = self.binary_train(X, y_dict)
        self.show_report(stat_d)

    def finetune(self, n):
        import pickle
        # n = 2
        # openfile = open('f1_turn{}.pk'.format(n),'rb')

        all_statistic = []
        X, Y, data, state_his_s, labels, uss = get_X_Y_from_raw_text()
        # turn_d = dict()
        # turn_d =pickle.load(openfile)
        # openfile.close()

        # for turn_n in [n]:
        #     turn_d[turn_n] = []
        for turn_n in [n]:

            finish = []
            text_file = open('f1_turn{}.txt'.format(n), 'r')
            lines = text_file.readlines()
            text_file.close()
            for line in lines:
                line = line.strip('\n')
                turn, hidden_size, rnn, action, f1 = line.split(',')
                turn = int(turn)
                hidden_size = int(hidden_size)
                rnn = int(rnn)
                if [turn, hidden_size, rnn] not in finish:
                    finish.append([turn, hidden_size, rnn])

            self.turn_num = turn_n
            new_data = self.history_padding(state_his_s, self.X_feature_number)
            data = [d for d in zip(new_data, labels) if d[1] != []]
            Y = [d[1] for d in data]
            X = [d[0] for d in data]
            y_dict = self.get_action_label_dict(Y)
            grid = []
            for hidden_size in self.hidden_size_range:
                for rnn in self.rnn_size_range:
                    grid.append([turn_n, hidden_size, rnn])

            for config in grid:
                contain = False
                turn_n, hidden_size, rnn = config
                if config in finish:
                    contain = True

                if contain:
                    print('already got train ', config)
                    continue

                # for r in turn_d[self.turn_num]:
                #     if self.hidden_size == r[0] and self.rnn_hidden_size == r[2]:
                #         contain = True
                # if contain:
                #     print('already got train ',r[0],r[1])
                #     continue
                self.hidden_size = hidden_size
                self.rnn_hidden_size = rnn
                self.build_binary_model()
                stat_d = self.binary_tuning(X, y_dict)
                # all_statistic.append([self.turn_num,self.hidden_size,self.rnn_hidden_size,stat_d])
                # turn_d[self.turn_num].append([self.hidden_size,self.rnn_hidden_size,stat_d])
                print('finish {},{},{}'.format(self.turn_num, self.hidden_size, self.rnn_hidden_size))
                # file = open('f1_turn{}.pk'.format(n), 'wb')
                file = open('f1_turn{}.txt'.format(n), 'a+')
                # pickle.dump(turn_d,file)
                for action in self.all_action:
                    if action in stat_d.keys():
                        config = [str(c) for c in config]
                        file.write(','.join(config + [action, str(stat_d[action])]) + "\n")
                file.close()

    def draw(self, data, action, n):
        import seaborn as sns
        import pandas as pd
        import matplotlib.pyplot as plt
        sns.set()
        data = pd.DataFrame(data, columns=self.rnn_size_range, index=self.hidden_size_range)
        ax = sns.heatmap(data, annot=True)
        plt.xlabel('rnn hidden unit')
        plt.ylabel('dense hidden unit')
        # 添加标题
        plt.title('heatmap turn:{} ,for {} ')
        plt.savefig('turn{}_action{}.png'.format(str(n), action), dpi=300)
        plt.show()

    def read_stat(self, n):
        # file = open('f1_turn3.pk', 'rb')
        # import pickle
        # turn_d = pickle.load(file)

        text_file = open('f1_turn{}.txt'.format(n), 'r')
        lines = text_file.readlines()
        text_file.close()

        action_dict = dict()
        for action in self.all_action:
            action_dict[action] = []

        for line in lines:
            line = line.strip('\n')
            turn, hidden_size, rnn, action, f1 = line.split(',')
            turn = int(turn)
            hidden_size = int(hidden_size)
            rnn = int(rnn)
            action_dict[action].append(float(f1))

        # for turn and each action ,draw heat map
        # turn = 3
        # cur = turn_d[turn]

        for action in self.all_action:
            if 'ask' in action:
                continue
            print(action)
            # action_arr = [u[-1][action] for u in cur]
            # action_arr_extend = [[u[0],u[1],u[-1][action]] for u in cur]
            data = np.array(action_dict[action]).reshape((len(self.hidden_size_range), len(self.rnn_size_range)))
            self.draw(data, action, n)


class DuelRnnCamRest(RnnBinaryModel):
    def get_utterance(self, history):
        self.build_counter(history)
        maxlen = 0
        print('get_w2i')
        his_data = []
        his_i = 0
        for turn in history:
            turn_vector = []
            for utterance in turn:
                words = nltk.word_tokenize(utterance.lower())
                seqs = self.words2seq(words, self.word2index)
                if len(seqs) > self.MAX_SENTENCE_LENGTH:
                    seqs = seqs[: self.MAX_SENTENCE_LENGTH]
                else:
                    seqs = [0] * (self.MAX_SENTENCE_LENGTH - len(seqs)) + seqs
            turn_vector += seqs
            turn_vector = self.fix_length_vector(turn_vector, self.MAX_SENTENCE_LENGTH * self.turn_number)
            his_data.append(turn_vector)
            his_i += 1
        his_data_array = np.array(his_data)
        # his = his.reshape((-1,self.turn_number* self.MAX_SENTENCE_LENGTH,))
        # Split input into training and test
        print('finish word 2 index')
        return his_data_array

    def default_config(self):
        self.is_binary_model = True
        self.max_vocab_size = 1200
        self.model = None
        self.hidden_size = 250
        self.turn_number = 3
        self.turn_num = 3
        self.MAX_SENTENCE_LENGTH = 30
        self.vocab_size, self.embedding_size = 1000, 100
        self.input_shape = (9,)
        self.batch_size = 100
        self.num_epochs = 50
        self.X_feature_number = 9
        self.model_path = "D://model/policy_learner_single.h5"
        more_actions = ['action_goodbye', 'action_morehelp', 'action_inform_address', \
                        'action_inform_food', 'action_inform_phone', \
                        'action_inform_area', 'action_inform_postcode', 'action_search_rest']
        self.all_action = more_actions
        self.set_slot()
        self.allow_action = self.all_action

    def build_binary_model(self):
        n_class = len(self.all_action)
        X, Y, data, state_his_s, labels, history_utterances = get_X_Y_from_raw_text()
        self.build_counter(history_utterances)
        use_pretrain = True
        word_index = self.word2index
        EMBEDDING_DIM = 100
        # x_utterance_concat = Input(shape =(self.MAX_SENTENCE_LENGTH * self.turn_number,), name="concat_utter")
        #### old impliment
        x_utterance_concat = Input(shape=(self.turn_number * self.MAX_SENTENCE_LENGTH,), name="concat_utter")
        if use_pretrain:
            embedding_matrix = self.get_embed_matrix(word_index, EMBEDDING_DIM)
            pretrain_embedding = Embedding(len(word_index) + 1,
                                           EMBEDDING_DIM,
                                           weights=[embedding_matrix],
                                           input_length=self.MAX_SENTENCE_LENGTH * self.turn_number,
                                           trainable=False)
            x_embed = pretrain_embedding(x_utterance_concat)

        else:
            x_embed = Embedding(self.vocab_size, self.embedding_size,
                                input_length=self.MAX_SENTENCE_LENGTH * self.turn_number)(x_utterance_concat)
        inputs_drop = SpatialDropout1D(0.2)(x_embed)
        word_level_rnn = Bidirectional(
            LSTM(self.hidden_size, dropout=0.1, recurrent_dropout=0.1, name='RNN',
                 return_sequences=True))(inputs_drop)
        rnn1_shape = Reshape((self.turn_number, self.MAX_SENTENCE_LENGTH, self.hidden_size * 2))(word_level_rnn)
        swap = Permute((1, 3, 2))(rnn1_shape)
        average = Lambda(self.reduce_mean)(swap)
        sentence_level_rnn = Bidirectional(
            LSTM(self.hidden_size, dropout=0.1, recurrent_dropout=0.1, name='RNN',
                 return_sequences=False))(average)

        binary = Dense(1, activation='sigmoid')(sentence_level_rnn)
        model = Model(inputs=x_utterance_concat, outputs=binary)
        model.compile(optimizer="adam",
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        self.model = model
        return model

    def binary_train(self, X, y_dict):
        stat_d = dict()
        for action in self.all_action:
            print('training ', action)
            X_, Y_ = self.downsample(X, y_dict[action])
            Xtrain, Xtest, ytrain, ytest = self.train_single_action(X_, Y_)
            y_hat = self.model.predict_on_batch(Xtest)
            y_hat = [y[0] for y in y_hat.tolist()]
            to_int = lambda x: 1 if x > 0.5 else 0
            y_hat = [to_int(y) for y in y_hat]
            stat = classification_report(ytest, y_hat)
            stat_d[action] = stat
            dir = self.model_path.split('.')[0]
            path = dir + action + '.' + self.model_path.split('.')[-1]
            self.save(path)
        return stat_d

    def load_data_and_train(self):
        X, Y, data, state_his_s, labels, uss = get_X_Y_from_raw_text()
        his_data_array = self.get_utterance(uss)
        # new_data = self.history_padding(uss, self.MAX_SENTENCE_LENGTH)
        data = [d for d in zip(his_data_array, labels) if d[1] != []]
        Y = [d[1] for d in data]
        X = [d[0] for d in data]

        y_dict = self.get_action_label_dict(Y)
        self.build_binary_model()
        stat_d = self.binary_train(X, y_dict)
        self.show_report(stat_d)


class MultiClassModel(ActionModel):
    def __init__(self, parameter=None):
        # self.__init__()
        if parameter == None:
            self.default_config()
            return

    def default_config(self):
        self.model = None
        self.hidden_size = 220
        self.input_shape = (59,)
        self.batch_size = 400
        self.num_epochs = 50
        self.X_feature_number = 9
        self.model_path = "D://model//policy_learner_kvret_new.h5"
        navigate = ['action_report_address', 'action_affirm_want_direction', 'action_report_traffic',
                    'action_report_distance', 'action_set_navigation']
        weather = ['action_ask_location', 'action_check_weather', 'action_clearify']
        schedule = ['action_report_event', 'action_set_reminder', 'action_ask_newevent_time']
        general = ['action_ok', 'action_morehelp', 'action_goodbye']
        all_action = navigate + weather + schedule + general
        self.all_action = all_action
        self.vocab_size = 500
        self.embedding_size = 200
        self.MAX_SENTENCE_LENGTH = 40
        self.turn_number = 4
        self.index2word = None
        self.word2index = None
        self.HIDDEN_LAYER_SIZE = 300
        self.max_vocab_size = 1300
        # keep it the last
        self.set_index_dict()

    def get_utterance(self, history):
        self.build_counter(history)
        maxlen = 0
        print('get_w2i')
        his_data = []
        his_i = 0
        for turn in history:
            turn_vector = []
            for utterance in turn:
                words = nltk.word_tokenize(utterance.lower())
                seqs = self.words2seq(words, self.word2index)
                if len(seqs) > self.MAX_SENTENCE_LENGTH:
                    seqs = seqs[: self.MAX_SENTENCE_LENGTH]
                else:
                    seqs = [0] * (self.MAX_SENTENCE_LENGTH - len(seqs)) + seqs
            turn_vector += seqs
            turn_vector = self.fix_length_vector(turn_vector, self.MAX_SENTENCE_LENGTH * self.turn_number)
            his_data.append(turn_vector)
            his_i += 1
        his_data_array = np.array(his_data)
        # his = his.reshape((-1,self.turn_number* self.MAX_SENTENCE_LENGTH,))
        # Split input into training and test
        print('finish word 2 index')
        return his_data_array

    def build_model(self):
        x = Input(shape=self.input_shape, name="input")
        print("Training...")
        layer1 = Dense(self.hidden_size, activation='relu')(x)
        layer2 = Dense(self.hidden_size, activation='relu')(layer1)
        layer2 = Dense(len(self.all_action), activation='softmax')(layer2)
        model = Model(inputs=x, outputs=layer2)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model

    def encoder(self, inputs_seqs, rnn_hidden_size, dropout_rate, rs):
        x_embed = Embedding(self.vocab_size, self.embedding_size,
                            input_length=self.MAX_SENTENCE_LENGTH * self.turn_number)(inputs_seqs)
        inputs_drop = SpatialDropout1D(0.2)(x_embed)
        encoded_Q = Bidirectional(
            LSTM(rnn_hidden_size, dropout=dropout_rate, recurrent_dropout=dropout_rate, name='RNN',
                 return_sequences=rs))(inputs_drop)
        return encoded_Q

    def build_utterance_model(self):
        MAX_SENTENCE_LENGTH = 40
        HIDDEN_LAYER_SIZE = 200
        x = Input(shape=self.input_shape, name="slot_vec")
        x_utterance_concat = Input(shape=(MAX_SENTENCE_LENGTH * self.turn_number,), name="concat_utter")
        encoded_utter = self.encoder(x_utterance_concat, HIDDEN_LAYER_SIZE, 0.1, False)
        print("Training utterance model...")
        layer1 = Dense(self.hidden_size, activation='relu')(x)
        layer2 = Dense(self.hidden_size, activation='relu')(layer1)
        slot_utter = Lambda(self.concat, name='combine')([layer2, encoded_utter])
        class_layer = Dense(len(self.all_action), activation='softmax')(slot_utter)

        model = Model(inputs=[x, x_utterance_concat], outputs=class_layer)
        # model = Model(inputs = x_utterance_concat, outputs = class_layer)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model

    # glove字典，每个单词对应一个100维的向量
    def get_glove_dict(self):
        embeddings_index = {}
        # f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
        f = open('D://udc//data//glove.6B.100d.txt', encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Found %s word vectors.' % len(embeddings_index))
        return embeddings_index

    def build_utterance_model2(self):
        MAX_SENTENCE_LENGTH = 40
        HIDDEN_LAYER_SIZE = 200
        x_utterance_concat = Input(shape=(MAX_SENTENCE_LENGTH * self.turn_number,), name="concat_utter")
        encoded_utter = self.encoder(x_utterance_concat, HIDDEN_LAYER_SIZE, 0.1, False)
        print("building utterance model...")
        class_layer = Dense(len(self.all_action), activation='softmax')(encoded_utter)
        model = Model(inputs=x_utterance_concat, outputs=class_layer)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model

    def build_tokenizer(self):
        X, Y, history_utterances = self.load_data_from_pk()
        us = []
        for h in history_utterances:
            us.extend(h)
        us = list(set(us))
        # texts = self.get_text(history_utterances)
        texts = us
        tokenizer = Tokenizer(num_words=self.max_vocab_size)
        tokenizer.fit_on_texts(texts)
        self.tokenizer = tokenizer
        return tokenizer

    # def get_embed_matrix(self, word_index, EMBEDDING_DIM):
    #     GLOVE_DIR = 'text_data/glove.6B'
    #     embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    #     embedding_index = self.get_glove_dict()
    #     for word, i in word_index.items():
    #         embedding_vector = embedding_index.get(word)
    #         if embedding_vector is not None:
    #             # words not found in embedding index will be all-zeros.
    #             embedding_matrix[i] = embedding_vector
    #     return embedding_matrix

    def build_pretrain_model(self):
        self.build_tokenizer()
        # sequences = self.tokenizer.texts_to_sequences(texts
        # word_index = self.tokenizer.word_index
        word_index = self.word2index
        EMBEDDING_DIM = 100
        embedding_matrix = self.get_embed_matrix(word_index, EMBEDDING_DIM)
        embedding_layer = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=self.MAX_SENTENCE_LENGTH * self.turn_number,
                                    trainable=False)

        x_utterance_concat = Input(shape=(self.MAX_SENTENCE_LENGTH * self.turn_number,), name="concat_utter")
        embedded_sequences = embedding_layer(x_utterance_concat)
        inputs_drop = SpatialDropout1D(0.2)(embedded_sequences)
        encoded_Q = Bidirectional(
            LSTM(self.hidden_size, dropout=0.1, recurrent_dropout=0.1, name='RNN',
                 return_sequences=False))(inputs_drop)
        class_layer = Dense(len(self.all_action), activation='softmax')(encoded_Q)
        model = Model(inputs=x_utterance_concat, outputs=class_layer)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model

    def load_data_from_pk(self):
        all_intent = ['navigate', 'weather', 'schedule']
        all_slot = ["room",
                    "agenda",
                    "time",
                    "date",
                    "party",
                    "event",
                    "weather_attribute",
                    "date",
                    "location",
                    "distance",
                    "traffic_info",
                    "poi_type",
                    "address",
                    "poi"]
        num_slot = len(all_slot)
        import pickle
        fw = open('C://starter-pack-rasa-stack-master//starter-pack-rasa-stack-master//training_data2.pk', 'rb')
        data = pickle.load(fw)
        fw.close()
        X, Y = [], []
        history_utterances = []
        y_indexs = []
        for line in data:
            history_slot, history_utterance, intent, ys = line
            h_vec = []
            for h in history_slot:
                h_vec_tmp = [0] * num_slot
                for slot in h:
                    h_vec_tmp[all_slot.index(slot)] = 1
                h_vec += h_vec_tmp
            padding_vec = [0] * (4 * num_slot - len(h_vec)) + h_vec
            intent_vec = [0, 0, 0]
            intent_vec[all_intent.index(intent)] = 1
            x = padding_vec + intent_vec
            if len(x) != 59: continue
            # print('len x',len(x))
            y_ = [0] * len(self.all_action)
            if len(ys) < 1: continue
            y_index = self.all_action.index(ys[0])
            y_[self.all_action.index(ys[0])] = 1
            X.append(x)
            Y.append(y_)
            y_indexs.append(y_index)
            history_utterances.append(history_utterance)
        return X, Y, history_utterances

    def load_data_and_train(self):
        X, Y, history_utterances = self.load_data_from_pk()
        self.build_counter(history_utterances)
        # self.build_utterance_model2()
        print('len word2index', len(self.word2index))
        self.build_pretrain_model()
        # self.train(X,history_utterances,Y)
        # self.tokenizer
        self.train_from_utters(history_utterances, Y)
        self.save()

    def set_index_dict(self):
        X, Y, history = self.load_data_from_pk()
        last_utterances = [his[-1] for his in history]
        self.set_word_index_from_text(last_utterances)
        return

    # def words2seq(self, words, word2index):
    #     seqs = []
    #     for word in words:
    #         if word in word2index.keys():
    #             seqs.append(word2index[word])
    #         else:
    #             seqs.append(word2index["UNK"])
    #     return seqs

    def load_data_and_evaluate(self):
        X, Y, history_utterances = self.load_data_from_pk()
        print('model path is {}'.format(self.model_path))
        self.load_model(self.model_path)
        X = np.array(X, )  # .reshape((-1,59))
        Y = np.array(Y)
        his_data_array = self.get_utterance(history_utterances)
        print('x ', len(X))
        print('get_his_data_array')
        Xtrain, Xtest, his_train, his_test, ytrain, ytest = train_test_split(X, his_data_array, Y, test_size=0.1,
                                                                             random_state=42)
        self.evaluate([Xtest, his_test], ytest)

    def get_seq_by_tokenizer(self, history):
        print('get_seq by tokenizer')
        self.build_tokenizer()
        his_data = []
        his_i = 0
        for turn in history:
            # turn_vector = []
            seqs = self.tokenizer.texts_to_sequences(turn)
            vecs = pad_sequences(seqs, maxlen=self.MAX_SENTENCE_LENGTH)
            # vecs.flatten().tolist()
            # turn_vector += seqs
            turn_vector = self.fix_length_vector(vecs.flatten().tolist(), self.MAX_SENTENCE_LENGTH * self.turn_number)
            his_data.append(turn_vector)
            his_i += 1
        his_data_array = np.array(his_data)
        return his_data_array

    def train(self, X, history, Y):
        X = np.array(X, )  # .reshape((-1,59))
        X = np.zeros(X.shape)
        Y = np.array(Y)
        his_data_array = self.get_utterance(history)
        print('x ', len(X))
        print('get_his_data_array')
        Xtrain, Xtest, his_train, his_test, ytrain, ytest = train_test_split(X, his_data_array, Y, test_size=0.1,
                                                                             random_state=42)

        self.model.fit([Xtrain, his_train], ytrain, batch_size=self.batch_size, verbose=self.verbose,
                       epochs=self.num_epochs,
                       validation_data=([Xtest, his_test], ytest))
        self.evaluate([Xtest, his_test], ytest)

    def train_from_utters(self, history, Y):
        Y = np.array(Y)
        his_data_array = self.get_utterance(history)
        # his_data_array = self.get_seq_by_tokenizer(history)
        print('get_seq_by_tokenizer')
        his_train, his_test, ytrain, ytest = train_test_split(his_data_array, Y, test_size=0.1,
                                                              random_state=42)
        self.model.fit(his_train, ytrain, batch_size=self.batch_size,
                       epochs=self.num_epochs,
                       validation_data=(his_test, ytest), verbose=self.verbose)
        self.evaluate(his_test, ytest)

    def evaluate(self, Xtest, ytest):
        y_hat = self.model.predict_on_batch(Xtest)
        index_y_hat = [np.argmax(y_row) for y_row in y_hat]
        index_y = [np.argmax(y_row) for y_row in ytest]
        from sklearn.metrics import classification_report
        report = classification_report(index_y, index_y_hat, target_names=self.all_action)
        print(report)

    def utters2vec(self, us):
        turn_vector = []
        for u in us:
            words = nltk.word_tokenize(u.lower())
            word_ids = self.words2seq(words, self.word2index)
            sent_vec = self.fix_length_vector(word_ids, self.MAX_SENTENCE_LENGTH)
            turn_vector += sent_vec
        turn_vector = self.fix_length_vector(turn_vector, self.MAX_SENTENCE_LENGTH * self.turn_number)
        return turn_vector

    def get_next_action_from_utters(self, utters):
        vec = self.utters2vec(utters)
        vec = np.array(vec).reshape((1, self.MAX_SENTENCE_LENGTH * self.turn_number))
        prediction = self.model.predict_on_batch(vec)
        multi_action_prop_dict = dict()
        for i, action in enumerate(self.all_action):
            prop = prediction[0][i]
            # print(action,prop)
            multi_action_prop_dict[action] = float(prop)
        # print(multi_action_prop_dict)
        best_action = self.get_best_action(multi_action_prop_dict)
        print(best_action)
        return best_action


class DoubleRnn(MultiClassModel):

    def encoder(self, inputs_seqs, rnn_hidden_size, dropout_rate, rs):
        x_embed = Embedding(self.vocab_size, self.embedding_size, input_length=self.MAX_SENTENCE_LENGTH)(inputs_seqs)
        inputs_drop = SpatialDropout1D(0.2)(x_embed)
        encoded_Q = Bidirectional(
            LSTM(rnn_hidden_size, dropout=dropout_rate, recurrent_dropout=dropout_rate, name='RNN',
                 return_sequences=rs))(inputs_drop)
        return encoded_Q

    def build_utterance_model(self):
        MAX_SENTENCE_LENGTH = 40
        HIDDEN_LAYER_SIZE = 200
        x_utterance_concat = Input(shape=(self.turn_number, MAX_SENTENCE_LENGTH), name="concat_utter")
        encoded_utter = self.encoder(x_utterance_concat, HIDDEN_LAYER_SIZE, 0.1, False)
        encoded_utter2 = self.encoder(encoded_utter, HIDDEN_LAYER_SIZE, 0.1, False)
        print("building utterance model...")
        class_layer = Dense(len(self.all_action), activation='softmax')(encoded_utter2)
        model = Model(inputs=x_utterance_concat, outputs=class_layer)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model


class TfRnnModel(MultiClassModel):
    def __init__(self, parameter=None):
        # self.__init__()
        if parameter == None:
            self.default_config()
            return

    def RNN(self, X, weights, biases):
        # basic LSTM Cell.
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True)
        else:
            cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
        # lstm cell is divided into two parts (c_state, h_state)
        init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell, X, initial_state=init_state, time_major=False)
        tp = tf.transpose(outputs, [1, 0, 2])
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            outputs = tf.unpack(tp)  # states is the last outputs
        else:
            outputs = tf.unstack(tp)
        results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # shape = (128, 10)
        return results

    def get_batch(self, data, step, batch_size):
        data = data[step * batch_size:(step + 1) * batch_size]
        return [u[1] for u in data], [u[0] for u in data]

    def get_var(self):
        tf.set_random_seed(1)
        lr = 0.001
        # Define weights
        weights = {
            # (28, 128)
            'in': tf.Variable(tf.random_normal([self.embedding_size, self.hidden_size])),
            # (128, 10)
            'out': tf.Variable(tf.random_normal([self.hidden_size, len(self.all_action)]))
        }
        biases = {
            # (128, )
            'in': tf.Variable(tf.constant(0.1, shape=[self.hidden_size, ])),
            # (10, )
            'out': tf.Variable(tf.constant(0.1, shape=[len(self.all_action), ]))
        }
        # tf Graph input
        x = tf.placeholder(tf.int32, [None, self.MAX_SENTENCE_LENGTH * self.turn_number])
        W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name="W")
        y = tf.placeholder(tf.float32, [None, len(self.all_action)])
        return lr, weights, biases, x, W, y

    def reduce_mean(self, X):
        return K.mean(X, axis=-1)

    def tf_train(self, x, y, his_train, his_test, ytrain, ytest, train_op, accuracy):

        # his_train, his_test, ytrain, ytest
        data = [[y, x] for y, x in zip(ytrain, his_train)]
        with tf.Session() as sess:
            # tf.initialize_all_variables() no long valid from
            # 2017-03-02 if using tensorflow >= 0.12
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                init = tf.initialize_all_variables()
            else:
                init = tf.global_variables_initializer()
            sess.run(init)
            step = 0
            time = 0
            training_iters = len(ytrain)  # ((step + 2) * self.batch_size) < training_iters
            max_step = int(training_iters / self.batch_size)
            while time < 800:
                # print('{},{},{},{}'.format(step,batch_size,training_iters,(step+1) * batch_size ))
                batch_xs, batch_ys = self.get_batch(data, step, self.batch_size)
                sess.run([train_op], feed_dict={
                    x: batch_xs,
                    y: batch_ys,
                })
                if time % 8 == 0 or time > 790:
                    print(time // (max_step - 2))
                    print((time) * self.batch_size, sess.run(accuracy, feed_dict={
                        x: his_test[-self.batch_size:],
                        y: ytest[-self.batch_size:],
                    }))
                step = (step + 1) % (max_step - 2)
                time += 1

    def build_model(self):
        # set random seed for comparing the two result calculations
        lr, weights, biases, x, W, y = self.get_var()
        embedded_chars = tf.nn.embedding_lookup(W, x)
        pred = self.RNN(embedded_chars, weights, biases)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        train_op = tf.train.AdamOptimizer(lr).minimize(cost)
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        X, Y, history_utterances = self.load_data_from_pk()
        self.build_counter(history_utterances)
        Y = np.array(Y)
        his_data_array = self.get_utterance(history_utterances)
        data = []
        for i in range(len(Y)):
            data.append([Y[i], his_data_array[i]])

        # data = zip(Y,history_utterances)

    def double_rnn_tf(self):
        X, Y, history_utterances = self.load_data_from_pk()
        self.build_counter(history_utterances)
        Y = np.array(Y)
        his_data_array = self.get_utterance(history_utterances)
        data = []
        for i in range(len(Y)):
            data.append([Y[i], his_data_array[i]])

        lr, weights, biases, x, W, y = self.get_var()
        embedded_chars = tf.nn.embedding_lookup(W, x)
        # ?,160,embedding
        add_dim = tf.reshape(embedded_chars, (-1, self.turn_number, self.MAX_SENTENCE_LENGTH, self.embedding_size))
        # (?,4,40,200)
        # trans = tf.transpose(add_dim, perm=[0, ])
        turn = dict()
        # basic LSTM Cell.
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True, name='word_RNN')
        else:
            cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, name='word_RNN')
        for i in range(4):
            turn[i] = add_dim[:, i, :, :]
            turn[i] = tf.reshape(turn[i], (-1, self.MAX_SENTENCE_LENGTH, self.embedding_size))
            init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
            outputs, final_state = tf.nn.dynamic_rnn(cell, turn[i], initial_state=init_state, time_major=False)
            tp = tf.transpose(outputs, [1, 0, 2])
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                outputs = tf.unpack(tp)  # states is the last outputs
            else:
                outputs = tf.unstack(tp)
            turn[i] = outputs[-1]  #

        for t in range(4):
            turn[t] = tf.reshape(turn[t], (-1, 1, self.hidden_size))
        turns = tf.concat([turn[0], turn[1], turn[2], turn[3]], axis=1)
        # basic LSTM Cell.
        with tf.variable_scope('sentence_RNN'):
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True,
                                                    name='sentence_RNN')
            else:
                cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, name='sentence_RNN')
            # lstm cell is divided into two parts (c_state, h_state)
            init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
            outputs, final_state = tf.nn.dynamic_rnn(cell, turns, initial_state=init_state, time_major=False)
            tp = tf.transpose(outputs, [1, 0, 2])
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                outputs = tf.unpack(tp)  # states is the last outputs
            else:
                outputs = tf.unstack(tp)
            rnn_output = outputs[-1]  #
        pred = tf.matmul(rnn_output, weights['out']) + biases['out']  # shape = (128, 10)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        train_op = tf.train.AdamOptimizer(lr).minimize(cost)
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        his_train, his_test, ytrain, ytest = train_test_split(his_data_array, Y, test_size=0.1,
                                                              random_state=42)
        self.tf_train(x, y, his_train, his_test, ytrain, ytest, train_op, accuracy)

        # return pred,train_op,cost,correct_pred,accuracy

    def test_double_rnn_keras(self):
        n_class = len(self.all_action)
        X, Y, history_utterances = self.load_data_from_pk()
        self.build_counter(history_utterances)
        Y = np.array(Y)
        his_data_array = self.get_utterance(history_utterances)
        data = []
        for i in range(len(Y)):
            data.append([Y[i], his_data_array[i]])
        use_pretrain = True
        word_index = self.word2index
        EMBEDDING_DIM = 100
        x_utterance_concat = Input(shape=(self.turn_number * self.MAX_SENTENCE_LENGTH,), name="concat_utter")
        x_utterance_concat2 = Reshape((self.turn_number, self.MAX_SENTENCE_LENGTH,), name='to_turn_length')(
            x_utterance_concat)
        if use_pretrain:
            embedding_matrix = self.get_embed_matrix(word_index, EMBEDDING_DIM)
            pretrain_embedding = Embedding(len(word_index) + 1,
                                           EMBEDDING_DIM,
                                           weights=[embedding_matrix],
                                           input_length=(self.turn_number, self.MAX_SENTENCE_LENGTH),
                                           trainable=False)
            x_embed = pretrain_embedding(x_utterance_concat2)

        else:
            x_embed = Embedding(self.vocab_size, self.embedding_size,
                                input_length=(self.MAX_SENTENCE_LENGTH, self.turn_number))(x_utterance_concat)

        # x embed ? 4 40 100
        def slice(x, turn):
            """ Define a tensor slice function
            """
            return x[:, turn, :, :]

        turn = dict()
        turn[0] = Lambda(slice, arguments={'turn': 0})(x_embed)
        turn[1] = Lambda(slice, arguments={'turn': 1})(x_embed)
        turn[2] = Lambda(slice, arguments={'turn': 2})(x_embed)
        turn[3] = Lambda(slice, arguments={'turn': 3})(x_embed)
        word_level_rnns = dict()
        for i in range(4):
            x_embed = turn[i]
            x_embed = Reshape((self.MAX_SENTENCE_LENGTH, 100))(x_embed)
            inputs_drop = SpatialDropout1D(0.2)(x_embed)

            word_level_rnn = Bidirectional(
                LSTM(self.hidden_size, dropout=0.1, recurrent_dropout=0.1, name='RNN',
                     return_sequences=False))(inputs_drop)
            word_level_rnns[i] = word_level_rnn
        for i in range(4):
            word_level_rnns[i] = Reshape((1, 2 * self.hidden_size), name='add_dimension' + str(i))(word_level_rnns[i])
        turn_combined = Lambda(K.concatenate, arguments={'axis': 1})(
            [word_level_rnns[0], word_level_rnns[1], word_level_rnns[2], word_level_rnns[3]])
        sentence_level_rnn = Bidirectional(
            LSTM(self.hidden_size, dropout=0.1, recurrent_dropout=0.1, name='RNN',
                 return_sequences=False))(turn_combined)
        class_layer = Dense(n_class, activation='softmax')(sentence_level_rnn)
        model = Model(inputs=x_utterance_concat, outputs=class_layer)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        his_train, his_test, ytrain, ytest = train_test_split(his_data_array, Y, test_size=0.1,
                                                              random_state=42)
        self.model = model
        self.model.fit(his_train, ytrain, batch_size=self.batch_size,
                       epochs=self.num_epochs,
                       validation_data=(his_test, ytest), verbose=self.verbose)
        self.evaluate(his_test, ytest)

    def test_keras(self):
        n_class = len(self.all_action)
        X, Y, history_utterances = self.load_data_from_pk()
        self.build_counter(history_utterances)
        Y = np.array(Y)
        his_data_array = self.get_utterance(history_utterances)
        data = []
        for i in range(len(Y)):
            data.append([Y[i], his_data_array[i]])

        use_pretrain = True
        word_index = self.word2index
        EMBEDDING_DIM = 100

        # x_utterance_concat = Input(shape =(self.MAX_SENTENCE_LENGTH * self.turn_number,), name="concat_utter")

        #### old impliment
        x_utterance_concat = Input(shape=(self.turn_number * self.MAX_SENTENCE_LENGTH,), name="concat_utter")
        if use_pretrain:
            embedding_matrix = self.get_embed_matrix(word_index, EMBEDDING_DIM)
            pretrain_embedding = Embedding(len(word_index) + 1,
                                           EMBEDDING_DIM,
                                           weights=[embedding_matrix],
                                           input_length=self.MAX_SENTENCE_LENGTH * self.turn_number,
                                           trainable=False)
            x_embed = pretrain_embedding(x_utterance_concat)

        else:
            x_embed = Embedding(self.vocab_size, self.embedding_size,
                                input_length=self.MAX_SENTENCE_LENGTH * self.turn_number)(x_utterance_concat)
        inputs_drop = SpatialDropout1D(0.2)(x_embed)
        word_level_rnn = Bidirectional(
            LSTM(self.hidden_size, dropout=0.1, recurrent_dropout=0.1, name='RNN',
                 return_sequences=True))(inputs_drop)
        rnn1_shape = Reshape((self.turn_number, self.MAX_SENTENCE_LENGTH, self.hidden_size * 2))(word_level_rnn)
        swap = Permute((1, 3, 2))(rnn1_shape)
        average = Lambda(self.reduce_mean)(swap)
        sentence_level_rnn = Bidirectional(
            LSTM(self.hidden_size, dropout=0.1, recurrent_dropout=0.1, name='RNN',
                 return_sequences=False))(average)
        class_layer = Dense(n_class, activation='softmax')(sentence_level_rnn)
        model = Model(inputs=x_utterance_concat, outputs=class_layer)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.fit(his_data_array[:-100], Y[:-100], batch_size=80,
        #           epochs=7,
        #           validation_data=(his_data_array[-100:], Y[-100:]))
        his_train, his_test, ytrain, ytest = train_test_split(his_data_array, Y, test_size=0.1,
                                                              random_state=42)
        self.model = model
        self.model.fit(his_train, ytrain, batch_size=self.batch_size,
                       epochs=self.num_epochs,
                       validation_data=(his_test, ytest), verbose=self.verbose)
        self.evaluate(his_test, ytest)


class HybridDuelRnnModel(DuelRnnCamRest):
    def get_keras_hybird(self):
        use_pretrain = True
        word_index = self.word2index
        EMBEDDING_DIM = 100
        x_utterance_concat = Input(shape=(self.turn_number * self.MAX_SENTENCE_LENGTH,), name="concat_utter")
        slot_input = Input(shape=(self.turn_number , self.X_feature_number,), name="slot_input")
        x_utterance_concat2 = Reshape((self.turn_number, self.MAX_SENTENCE_LENGTH,), name='to_turn_length')(
            x_utterance_concat)
        if use_pretrain:
            embedding_matrix = self.get_embed_matrix(word_index, EMBEDDING_DIM)
            pretrain_embedding = Embedding(len(word_index) + 1,
                                           EMBEDDING_DIM,
                                           weights=[embedding_matrix],
                                           input_length=(self.turn_number, self.MAX_SENTENCE_LENGTH),
                                           trainable=False)
            x_embed = pretrain_embedding(x_utterance_concat2)

        else:
            x_embed = Embedding(self.vocab_size, self.embedding_size,
                                input_length=(self.MAX_SENTENCE_LENGTH, self.turn_number))(x_utterance_concat)

        # x embed ? 4 40 100
        def slice(x, turn):
            """ Define a tensor slice function
            """
            return x[:, turn, :, :]

        turn = dict()
        for t in range(self.turn_num):
            turn[t] =  Lambda(slice, arguments={'turn': t})(x_embed)

        word_level_rnns = dict()
        for i in range(self.turn_num):
            x_embed = turn[i]
            x_embed = Reshape((self.MAX_SENTENCE_LENGTH, 100))(x_embed)
            inputs_drop = SpatialDropout1D(0.2)(x_embed)

            word_level_rnn = Bidirectional(
                LSTM(self.hidden_size, dropout=0.1, recurrent_dropout=0.1, name='RNN',
                     return_sequences=False))(inputs_drop)
            word_level_rnns[i] = word_level_rnn
        for i in range(self.turn_num):
            word_level_rnns[i] = Reshape((1, 2 * self.hidden_size), name='add_dimension' + str(i))(word_level_rnns[i])
        all_turns  = [word_level_rnns[i] for i in range(self.turn_num)]
        turn_combined = Lambda(K.concatenate, arguments={'axis': 1})(all_turns)
        sentence_level_rnn = Bidirectional(
            LSTM(self.hidden_size, dropout=0.1, recurrent_dropout=0.1, name='RNN',
                 return_sequences=False))(turn_combined)        # slot_state
        slot_rnn = Bidirectional(
            LSTM(self.hidden_size, dropout=0.1, recurrent_dropout=0.1, name='slot_state_RNN',
                 return_sequences=False))(slot_input)
        hybird = Lambda(K.concatenate, arguments={'axis': 1})([sentence_level_rnn,slot_rnn])
        class_layer = Dense(1, activation='sigmoid')(hybird)
        model = Model(inputs=[x_utterance_concat,slot_input], outputs=class_layer)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


    # slot filling state + utterances
    def keras_hybird(self):
        n_class = 2
        X, Y, _, state_his_s, labels, uss = get_X_Y_from_raw_text()
        his_data_array = self.get_utterance(uss)
        # new_data = self.history_padding(uss, self.MAX_SENTENCE_LENGTH)
        data = []
        for i in range(len(labels)):
            slot, utt, label = state_his_s[i], his_data_array[i], labels[i]
            data.append([slot, utt, label])
        data = [d for d in data if d[2] != []]
        Y = [d[2] for d in data]
        his_data_array = [d[1] for d in data]
        slot_state_his = [d[0] for d in data]
        slot_state_his = self.history_padding(slot_state_his, self.X_feature_number)
        slot_state_his = [[[float(e) for e in d] for d in r] for r in slot_state_his]
        y_dict = self.get_action_label_dict(Y)


        for action in self.all_action:
            print('training ', action)
            X = [[utt,slot] for utt,slot in zip(his_data_array,slot_state_his)]
            # X_, Y_ = self.downsample(X, y_dict[action])
            #             # his_data_array = [x[0] for x in X_]
            #             # slot_state_his = [x[1] for x in X_]
            #             # Y = Y_
            Y = y_dict.get(action)

            his_train, his_test, ytrain, ytest, slot_train, slot_test = train_test_split(his_data_array, Y, slot_state_his,
                                                                                         test_size=0.1,
                                                                                         random_state=42)
            self.model = self.get_keras_hybird()
            self.model.fit([his_train,slot_train], ytrain, batch_size=self.batch_size,
                           epochs=10,
                           validation_data=([his_test,slot_test], ytest), verbose=2)
            y_hat = self.model.predict_on_batch([his_test,slot_test])
            y_hat = [y[0] for y in y_hat.tolist()]
            to_int = lambda x: 1 if x > 0.5 else 0
            y_hat = [to_int(y) for y in y_hat]
            stat = classification_report(ytest, y_hat)
            print(action,stat)

    def get_var(self):
        tf.set_random_seed(1)
        lr = 0.001
        # Define weights
        weights = {
            # (28, 128)
            'in': tf.Variable(tf.random_normal([self.embedding_size, self.hidden_size*2])),
            # (128, 10)
            'out': tf.Variable(tf.random_normal([self.hidden_size*2,2]))
        }
        biases = {
            # (128, )
            'in': tf.Variable(tf.constant(0.1, shape=[self.hidden_size*2, ])),
            # (10, )
            'out': tf.Variable(tf.constant(0.1, shape=[2, ]))
        }
        # tf Graph input
        x = tf.placeholder(tf.int32, [None, self.MAX_SENTENCE_LENGTH * self.turn_number])
        W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name="W")
        y = tf.placeholder(tf.float32, [None, 2])

        slot_state = tf.placeholder(tf.float32, [None,  self.turn_number,self.X_feature_number])

        return lr, weights, biases, x, W, y,slot_state

    def tf_train(self, x, y, slot,his_train, his_test,slot_train,slot_test, ytrain, ytest, train_op, accuracy):
        # his_train, his_test, ytrain, ytest

        data = []
        # slot,utt,label
        for i in range(len(ytrain)):
            slot_,utt,label = slot_train[i],his_train[i],ytrain[i]
            data.append([slot_,utt,label])
        with tf.Session() as sess:
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                init = tf.initialize_all_variables()
            else:
                init = tf.global_variables_initializer()
            sess.run(init)
            step = 0
            time = 0
            training_iters = len(ytrain)  # ((step + 2) * self.batch_size) < training_iters
            max_step = int(training_iters / self.batch_size)
            while time < 800:
                # print('{},{},{},{}'.format(step,batch_size,training_iters,(step+1) * batch_size ))
                batch_slots,batch_utts, batch_ys = self.get_batch(data, step, self.batch_size)
                batch_slots =np.array(batch_slots).reshape((-1,self.turn_num,self.X_feature_number))
                sess.run([train_op], feed_dict={
                    x: batch_utts,
                    y: batch_ys,
                    slot:batch_slots
                })
                if time % 8 == 0 or time > 790:
                    print(time // (max_step - 2))
                    print((time) * self.batch_size, sess.run(accuracy, feed_dict={
                        x: his_test[-self.batch_size:],
                        y: ytest[-self.batch_size:],
                        slot: slot_test[-self.batch_size:]
                    }))
                step = (step + 1) % (max_step - 2)
                time += 1


        return sess
    def get_batch(self, data, step, batch_size):
        data = data[step * batch_size:(step + 1) * batch_size]
        return [u[0] for u in data], [u[1] for u in data], [u[2] for u in data]

    def double_rnn_tf(self):
        X, Y, _, state_his_s, labels, uss = get_X_Y_from_raw_text()
        his_data_array = self.get_utterance(uss)
        # new_data = self.history_padding(uss, self.MAX_SENTENCE_LENGTH)

        data = []
        for i in range(len(labels)):
            slot,utt,label = state_his_s[i],his_data_array[i],labels[i]
            data.append([slot,utt,label])

        data = [d for d in data if d[2] != []]
        Y = [d[2] for d in data]
        his_data_array = [d[1] for d in data]
        slot_state_his = [d[0] for d in data]
        slot_state_his = self.history_padding(slot_state_his,self.X_feature_number)
        slot_state_his = [[[float(e) for e in d] for d in r] for r in slot_state_his]

        y_dict = self.get_action_label_dict(Y)
        Y = y_dict.get('action_search_rest')
        Y_ = []
        for y in Y:
            if y == 1:
                Y_.append([0,1])
            else:
                Y_.append([1,0])
        #Y = [[0,1] for y in Y if y == 0 else [1,0]]
        Y = Y_


        lr, weights, biases, x, W, y ,slot_state= self.get_var()
        embedded_chars = tf.nn.embedding_lookup(W, x)
        # ?,160,embedding
        add_dim = tf.reshape(embedded_chars, (-1, self.turn_number, self.MAX_SENTENCE_LENGTH, self.embedding_size))
        # (?,4,40,200)
        # trans = tf.transpose(add_dim, perm=[0, ])
        turn = dict()
        # basic LSTM Cell.
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True, name='word_RNN')
        else:
            cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, name='word_RNN')
        for i in range(self.turn_num):
            turn[i] = add_dim[:, i, :, :]
            turn[i] = tf.reshape(turn[i], (-1, self.MAX_SENTENCE_LENGTH, self.embedding_size))
            init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
            outputs, final_state = tf.nn.dynamic_rnn(cell, turn[i], initial_state=init_state, time_major=False)
            tp = tf.transpose(outputs, [1, 0, 2])
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                outputs = tf.unpack(tp)  # states is the last outputs
            else:
                outputs = tf.unstack(tp)
            turn[i] = outputs[-1]  #

        for t in range(self.turn_num):
            turn[t] = tf.reshape(turn[t], (-1, 1, self.hidden_size))
        all_turns = [turn[i] for i in range(self.turn_num)]
        turns = tf.concat(all_turns, axis=1)
        # basic LSTM Cell.
        with tf.variable_scope('sentence_RNN'):
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True,
                                                    name='sentence_RNN')
            else:
                cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, name='sentence_RNN')
            # lstm cell is divided into two parts (c_state, h_state)
            init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
            outputs, final_state = tf.nn.dynamic_rnn(cell, turns, initial_state=init_state, time_major=False)
            tp = tf.transpose(outputs, [1, 0, 2])
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                outputs = tf.unpack(tp)  # states is the last outputs
            else:
                outputs = tf.unstack(tp)
            rnn_output = outputs[-1]  #


        # slot model
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True, name='slot_RNN')
        else:
            cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, name='slot_RNN1')
        init_state = cell.zero_state(self.batch_size, dtype=tf.float32)
        slot_state_outputs, final_state = tf.nn.dynamic_rnn(cell,slot_state, initial_state=init_state, time_major=False)
        tp = tf.transpose(slot_state_outputs, [1, 0, 2])
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            outputs = tf.unpack(tp)  # states is the last outputs
        else:
            outputs = tf.unstack(tp)
        slot_state_halt = outputs[-1]  #
        hybird = tf.concat([rnn_output, slot_state_halt], axis=1)

        pred = tf.matmul(hybird, weights['out']) + biases['out']  # shape = (128, 10)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        train_op = tf.train.AdamOptimizer(lr).minimize(cost)
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        his_train, his_test, ytrain, ytest ,slot_train,slot_test= train_test_split(his_data_array, Y,slot_state_his, test_size=0.1,
                                                              random_state=42)
        sess = self.tf_train(x, y,slot_state, his_train, his_test, slot_train,slot_test,ytrain, ytest, train_op, accuracy)



# m = MultiClassModel()
# m.load_data_and_train()
# # m.load_data_from_pk()
# # # #m.set_index_dict()
# # #m.load_data_and_evaluate()
# m.load_data_and_train()
# binary = RnnBinaryModel()
# #binary.load_data_and_train()
# binary.finetune(4)
# binary.read_stat(2)
# binary.read_stat(3)

# tfmodel = TfRnnModel()
# tfmodel.double_rnn_tf()
# tfmodel.set_epoch(70)
# tfmodel.test_double_rnn_keras()

# duelrnn = DuelRnnCamRest()
# duelrnn.set_verbose(2)
# duelrnn.load_data_and_train()

hybird = HybridDuelRnnModel()
hybird.keras_hybird()

x = 1