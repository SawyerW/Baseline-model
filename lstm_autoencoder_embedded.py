#!/usr/bin/env python
import keras
import os
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import pandas as pd
import re
import csv
from keras.layers import Input, LSTM, RepeatVector,TimeDistributed,Dense,Dropout,Embedding
from keras.models import Model
from collections import defaultdict
from itertools import count
from functools import partial
from collections import defaultdict
from keras.models import Sequential
from keras.layers import LSTM
from collections import defaultdict
from keras.preprocessing import text
import nltk
import sys
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
import numpy as np
from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.preprocessing import Normalizer
from keras.optimizers import Adam, RMSprop, SGD
from Get_Amazon_reviews_vectors import write_vec_to_csv
reload(sys)
sys.setdefaultencoding('utf8')
def get_data():
    targets = ['cameras','laptops','mobilephone','tablets','TVs','video_surveillance']
    path =['/home/flippped/Desktop/xiangmu/baseline/Reviews/cameras',
           '/home/flippped/Desktop/xiangmu/baseline/Reviews/laptops',
           '/home/flippped/Desktop/xiangmu/baseline/Reviews/mobilephone',
           '/home/flippped/Desktop/xiangmu/baseline/Reviews/tablets',
           '/home/flippped/Desktop/xiangmu/baseline/Reviews/TVs',
           '/home/flippped/Desktop/xiangmu/baseline/Reviews/video_surveillance']
    data = []
    target=[]
    filename = []
    for j in  xrange(len(path)):

        print path[j]
        #for i in path[j]:
        for file in os.listdir(path[j]):

            with open(os.path.join(path[j], file), 'r') as f:
                 document = f.read().lower()
                 target.append(targets[j])
                 filename.append(file)
                 data.append(document)

    #print len(target)
    return data,target,filename
def preprocess(text):
    text = text.lower()
    doc = ' '.join(re.findall(r"[\w']+|[.,!?;/-]", text))
    #print doc
    doc = word_tokenize(doc)
    #doc = keras.preprocessing.text.Tokenizer(num_words=None,lower=True, split=" ").fit_on_texts(doc)
    #print doc
    #doc = [word for word in doc if word.isalpha()]
    #doc = [word for word in doc if word not in stop_words]
    #print doc
    return doc
def get_corpus():
    #stemmer = PorterStemmer()
    data,target,filename = get_data()
    # corpus_train_tmp = [preprocess(text) for text in data]
    corpus_train_tmp = data
    #filter empty docs
    corpus_train, data, target, filenames = filter_docs(corpus_train_tmp,data,target, filename, lambda doc: (len(doc) != 0))
    for i in corpus_train:
        print i
    return corpus_train, target, filenames
def filter_docs(corpus, texts, labels, filenames, condition_on_doc):
    """
    Filter corpus, texts and labels given the function condition_on_doc which takes
    a doc.
    The document doc is kept if condition_on_doc(doc) is true.
    """

    number_of_docs = len(corpus)

    if texts is not None:
        texts = [text for (text, doc) in zip(texts, corpus)
                 if condition_on_doc(doc)]

    labels = [i for (i, doc) in zip(labels, corpus) if condition_on_doc(doc)]
    corpus = [doc for doc in corpus if condition_on_doc(doc)]
    filenames = [filename for (filename,doc) in zip(filenames, corpus) if condition_on_doc(doc)]
    print("{} docs removed".format(number_of_docs - len(corpus)))

    return (corpus, texts, labels, filenames)
def string_to_integer(strings):
    string_to_number = {string: i for i, string in enumerate(set(strings), 1)}
    test = [(string_to_number[string], string) for string in strings]
    # string_to_number = defaultdict(partial(next, count(1)))
    # test = [(string_to_number[string], string) for string in strings]
    return test
def preprocess_embedding():
    corpus_train, target, filenames = get_corpus()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus_train)
    sequences = tokenizer.texts_to_sequences(corpus_train)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    MAX_SEQUENCE_LENGTH = 50
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('/home/flippped/Desktop/xiangmu/baseline/GoogleNews-vectors-negative300.bin.gz', binary=True)
    word2vec_model.init_sims(replace=True)

    # create one matrix for documents words
    EMBEDDING_DIM = 300
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    print embedding_matrix.shape
    for word, i in word_index.items():
            try:
                embedding_vector = word2vec_model[str(word)]
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector

            except:
                continue


    return data,target,filenames,embedding_matrix, word_index

def lstm():
    data,  targets, filenames, embedding_matrix, word_index = preprocess_embedding()
    EMBEDDING_DIM = 300
    MAX_SEQUENCE_LENGTH = 50
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable= False,
                                name='layer_embedding') #mask_zero=True,


    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    x1 = LSTM(150, return_sequences=True,name='lstm_1')(embedded_sequences)

    #x2 = LSTM(75, return_sequences=True,name='lstm_2')(x1)
    encoded = LSTM(30,name='lstm_3')(x1)
    x3 = RepeatVector(MAX_SEQUENCE_LENGTH,name='layer_repeat')(encoded)
   # x4 = LSTM(75, return_sequences=True,name='lstm_4')(x3)
    x5 = LSTM(150, return_sequences=True,name='lstm_5')(x3)
    decoded = LSTM(300, return_sequences=True,activation='linear',name='lstm_6')(x5)

    sequence_autoencoder = Model(sequence_input, decoded)
    #print sequence_autoencoder.get_layer('lstm_6').output
    encoder = Model(sequence_input, encoded)
    sequence_autoencoder.compile(loss='cosine_proximity',
                  optimizer='sgd')#, metrics=['acc'])
    embedding_layer = Model(inputs=sequence_autoencoder.input,
                                     outputs=sequence_autoencoder.get_layer('layer_embedding').output)


    sequence_autoencoder.fit(data, embedding_layer.predict(data), epochs=5)


    # for i in  sequence_autoencoder.layers[3].get_weights()[0]:
    #     print i
    #
    # print sequence_autoencoder.layers[3].get_weights()[0][1]

    # print sequence_autoencoder.layers[1].get_weights()[0][1].shape
    # print sequence_autoencoder.layers[2].get_weights()[0][1].shape
    # print sequence_autoencoder.layers[3].get_weights()[0][1].shape
    # print sequence_autoencoder.layers[4].get_weights()[0][1].shape
    # #print sequence_autoencoder.layers[5].get_weights()[0][1].shape
    # print sequence_autoencoder.layers[6].get_weights()[0][1].shape
    # print sequence_autoencoder.layers[7].get_weights()[0][1].shape

    csvname = 'lstm_autoencoder_weight'
    write_vec_to_csv(sequence_autoencoder.layers[3].get_weights()[0],targets,filenames,csvname)
    #sequence_autoencoder.save_weights('embedded_weights.h5')
    # model = Sequential()
    # time_steps = 50
    # input_size = 300
    # batch_size = 9
    # latent_dim = 10
    # #
    # # truncate and pad input sequences
    # max_review_length = 50
    # model.add(Embedding(input_dim=max_integer,output_dim=input_size,input_length=max_review_length,mask_zero=True,name='layer_embedding'))
    # print model.summary()
    # model.add((LSTM(150, return_sequences=True, input_shape=(time_steps,input_size),name='lstm_1')))
    # print model.summary()
    # model.add(LSTM(75, return_sequences=True,name='lstm_2' ))
    # model.add(LSTM(30,name='lstm_3'))
    # model.add(Dropout(0.5))
    #
    # model.add(RepeatVector(time_steps,name='layer_repeat'))
    # model.add(LSTM(75, return_sequences=True,stateful=False,name='lstm_4'))
    # model.add(LSTM(150,return_sequences=True,name='lstm_5'))
    # model.add(LSTM(300,return_sequences=True,activation='softmax',name='lstm_6'))
    #
    # #model.add()
    # print model.summary()
    #
    # model.compile(optimizer='rmsprop',
    #               loss='mse', )
    # model.fit(a, model.get_layer('lstm_6').output, nb_epoch=5)
    # model.save('test.h5')

def write_vec_to_csv(doc_vector_train,targets,filenames, csvname):
    # target_name_train = []
    # for i in xrange(len(targets)):
    #     target_name_train.append(newsgroups_train.target_names[newsgroups_train.target[i]])
    # print len(target_name_train)
    # print doc_vector_train_tsne.shape
    # print len(newsgroups_train.filenames)
    output_train = np.column_stack((targets,filenames, doc_vector_train))
    output_train = np.array(output_train)

    with open('reviews_50_' + csvname + '_.csv', 'w') as f:
        fieldnames = ['target_names', 'filenames']
        for i in xrange(len(doc_vector_train[1])):
            fieldnames.append('x'+ str(i))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        #writer = csv.DictWriter(f)
        writer.writeheader()
        writer = csv.writer(f)

        writer.writerows(output_train)

def main():
 lstm()
 #preprocess_embedding()
 #embedding()


if __name__ == "__main__":
 main()
