import nltk
from parameter import MAX_FEATURES,MAX_SENTENCE_LENGTH
import pandas as pd
from collections import Counter
def get_pair(number, dialogue):
    pairs = []
    for conversation in dialogue:
        utterances = conversation[2:].strip('\n').split('\t')
        # print(utterances)
        # break

        for i, utterance in enumerate(utterances):
            if i % 2 != 0: continue
            pairs.append([utterances[i], utterances[i + 1]])
            if len(pairs) >= number:
                return pairs
    return pairs


def convert_dialogue_to_pair(k):
    dialogue = open('dialogue_alibaba2.txt', encoding='utf-8', mode='r')
    dialogue = dialogue.readlines()
    dialogue = [p for p in dialogue if p.startswith('1')]
    print(len(dialogue))
    pairs = get_pair(k, dialogue)
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




def get_alibaba(N):
    df_sentiment = convert_dialogue_to_pair(N)
    #df_sentiment.to_csv('alibaba.csv',index=False,columns=['label','sentence_q','sentence_a'],encoding='utf-8')
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
            res.append([0, question, answer])
            # print('0')
        if label == '1':
            res.append([1, question, answer])
            # print('1')
    print('{} words in the data set'.format(vocab_size) )
    return res,vocab_size