# -*- coding: utf-8 -*-
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional
from keras.callbacks import EarlyStopping
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.models import load_model
from sklearn.model_selection import train_test_split
import collections
import pickle
import numpy as np
import jieba
import random
# import nltk
# from snownlp import SnowNLP

TEST_SEED = 3000
MAX_SENTENCE_LENGTH = 80
MAX_FEATURES = 2000   #2000
MAX_SENT = 15

def words_seg(sentence):
    # sentence = SnowNLP(sentence)                      # 使用SnoeNLP分詞
    # words = sentence.words  
    sentence = jieba.cut(sentence)                      # 使用Jieba分詞
    words = list(sentence)
    return words  

def train_lstm():
    global TEST_SEED, MAX_SENTENCE_LENGTH, MAX_FEATURES,MAX_SENT
    word_freqs = collections.Counter()
    maxlen = 0                                          # 計算訓練資料的字句最大字數
    num_recs = 0
    test_seed = TEST_SEED                                          # 取用學習素材筆樹
    print "preprocess positive data ..."
    cnt = 0
    with open('./positive.txt','r+') as f:
        for line in f:
            words = words_seg(line.decode('utf8'))
            if len(words) >= MAX_SENT:
                if len(words) > maxlen:
                    maxlen = len(words)
                for word in words:
                    word_freqs[word] += 1
                num_recs += 1
                cnt += 1
                if cnt % test_seed == 0:
                    break
    print "preprocess negative data ..."    
    cnt = 0    
    with open('./negative.txt','r+') as f:
        for line in f:
            words = words_seg(line.decode('utf8'))
            if len(words) >= MAX_SENT:
                if len(words) > maxlen:
                    maxlen = len(words)
                for word in words:
                    word_freqs[word] += 1
                num_recs += 1
                cnt += 1
                if cnt % test_seed == 0:
                    break
    print '-> num_recs:', num_recs
    print '-> max_len:', maxlen
    print '-> nb_words:', len(word_freqs)
    
    ## 準備數據
    # MAX_FEATURES = 2000     # int(len(word_freqs) * 0.7) 
    print '-> MAX_FEATURES:', MAX_FEATURES
    # MAX_SENTENCE_LENGTH = 80
    vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
    word_index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
    word_index["PAD"] = 0
    word_index["UNK"] = 1
    index2word = {v:k for k, v in word_index.items()}
    X = np.empty(num_recs,dtype=list)
    y = np.zeros(num_recs)
    i = 0
    cnt = 0
    # 讀取訓練資料，將每一單字以 dictionary 儲存
    print "process positive data..."
    with open('./positive.txt','r+') as f:
        for line in f:
            words = words_seg(line.decode('utf8'))
            if len(words) >= MAX_SENT:
                label = 1
                seqs = []
                for word in words:
                    if word in word_index:
                        seqs.append(word_index[word])
                    else:
                        seqs.append(word_index["UNK"])
                X[i] = seqs
                y[i] = label
                i += 1
                cnt += 1
                if cnt % test_seed == 0:
                    break
    print "process negative data..."
    cnt = 0
    with open('./negative.txt','r+') as f:
        for line in f:
            words = words_seg(line.decode('utf8'))
            if len(words) >= MAX_SENT:
                label = 0
                seqs = []
                for word in words:
                    if word in word_index:
                        seqs.append(word_index[word])
                    else:
                        seqs.append(word_index["UNK"])
                X[i] = seqs
                y[i] = label
                i += 1
                cnt += 1
                if cnt % test_seed == 0:            # 數量足夠
                    break

    # 字句長度不足補空白        
    X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
    # 資料劃分訓練組與測試資料
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=42)
    # 模型構建（LSTM）
    EMBEDDING_SIZE = 128
    HIDDEN_LAYER_SIZE = 64
    BATCH_SIZE = 32
    NUM_EPOCHS = 12
    model = Sequential()
    # 加嵌入層
    model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
    # 加雙向LSTM層
    model.add(Bidirectional(LSTM(HIDDEN_LAYER_SIZE, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), merge_mode='concat'))
    # 加LSTM層
    model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
    # model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))            # 'hard_sigmoid'
    model.summary()
    # binary_crossentropy:二分法
    model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
    # 宣告提早停止機制
    # es = EarlyStopping(monitor="val_acc", patience = 5, mode="auto" )
    # 模型訓練
    model.fit(Xtrain, 
              ytrain, 
              batch_size=BATCH_SIZE, 
              epochs=NUM_EPOCHS,
              callbacks=[EarlyStopping(monitor="val_acc", patience = 5, mode="auto" )],
              validation_data=(Xtest, ytest))
    return model, word_index        

def model_save(model, words_index):                     # 模型存檔
    model.save('lstm.h5')                               # creates a HDF5 file 'model.h5'
    with open('lstm_word_index.pickle', 'wb') as handle:
        pickle.dump(word_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

def model_load(h5_file, index_file):
    model = load_model(h5_file)                         # load lstm model                             
    with open(index_file, 'rb') as file:                # load words_index file
        words_index = pickle.load(file)   
    return model, words_index   

# def test_jieba():                                       # 測試Jieba分詞
#     INPUT_SENTENCES = [ u"""沒看卻能知道被嘴，幽默，而且也太自以為，被嗆個幾句在那邊鄭重道歉，我你老師笑到差點從椅子上掉下來""",
#                         u"""能這樣做已經很好了，不然妳想怎樣""",
#                         u"""好還要更好，一定要讓你知道""",
#                         u"""妳的好只有我知道""",
#                         u"""學點東西就出來現，鄙視你""",
#                         u"""妳只有這點能耐嗎？""",
#                         u"""不要再當酸民了，好嗎？""",
#                         u"""最漂亮的你，依舊動人可愛""",
#                         u"""妳想我嗎，我很想你""",
#                         u"""妳的自信讓我發光"""
#                     ]
#     for sentence in INPUT_SENTENCES:
#         jba_str = jieba.cut(sentence)  
#         # print type(jba_str)  
#         print list(jba_str)

def model_predict(model, word_index):                               # 預測模型
    cnt = 1
    INPUT_SENTENCES = []
    with open('./positive.txt','r+') as f:
        for line in f:         
            dst = random.randint(0, 50)
            if dst == 25:
                cnt += 1
                INPUT_SENTENCES.append(line.decode('utf8'))
            if cnt % 11 == 0:
                break
    cnt = 1            
    with open('./negative.txt','r+') as f:
        for line in f:      
            dst = random.randint(0, 50)
            if dst == 25:
                cnt += 1
                INPUT_SENTENCES.append(line.decode('utf8'))
            if cnt % 11 == 0:
                break                     
    XX = np.empty(len(INPUT_SENTENCES), dtype=list)
    i=0
    for sentence in  INPUT_SENTENCES:
        words = words_seg(sentence)
        seq = []
        for word in words:
            if word in word_index:
                # print word
                seq.append(word_index[word])
            else:
                seq.append(word_index['UNK'])
        XX[i] = seq
        i += 1
    seq_XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
    labels_vol = [round(x[0],4) for x in model.predict(seq_XX) ]
    # 歸納預測，並將結果四捨五入，轉換為 0 或 1
    labels = [int(round(x[0])) for x in model.predict(seq_XX) ]
    label2word = {1:u'正面', 0:u'負面'}
    # 顯示結果
    print u"顯示測試結果"
    print "============"
    print u"判斷", u"情感分數", u"句子"
    for i in range(len(INPUT_SENTENCES)):
        print label2word[labels[i]], labels_vol[i], INPUT_SENTENCES[i].replace('\n','',1)

if __name__ == '__main__':
    model, word_index = train_lstm()
    model_save(model, word_index)
    # model, word_index = model_load('lstm.h5','lstm_word_index.pickle')
    model_predict(model, word_index)
    # test_jieba()
    


