# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 18:46:15 2020

@author: Ines
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 18:46:15 2020

@author: Ines
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import re #library for using RegEx
import nltk #library for various NLP toolkits

import heapq

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score

# Splitting training and testing set
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB


# dataset = pd.read_excel(r'C:\Users\Ines\Desktop\Untitled Folder\hampirJadiHarusny\sentiment-dataset.xlsx')
dataset = pd.read_excel('sentiment-dataset.xlsx')

dataset.SENTIMEN = pd.Categorical(dataset.SENTIMEN) 
dataset['id_sentimen'] = dataset.SENTIMEN.cat.codes #save into new column
print(dataset.SENTIMEN)
print(dataset.id_sentimen)


from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
stopwords = StopWordRemoverFactory().get_stop_words()

corpus = []
for i in range(0,553):
    review = re.sub('[^a-zA-Z]', ' ', dataset['COMMENT'][i]) #dataset yg ad angka dihapus, yg disimpan dari A-Z yang gede kecil
    review = review.lower() #ubah jadi lowercase
    review = review.split() #split kalimat jadi kata-kata e.g.['aplikasi', 'keren', 'banget']
    factory = StemmerFactory() #punya si sastrawi
    stemmer = factory.create_stemmer() #siapin alat buat stemming
    review = [stemmer.stem(word) for word in review if not word in set(stopwords)] #untuk setiap kata di review kalau gak termasuk di stopword akan dicoba stemming
    review = ' '.join(review)
    corpus.append(review)
    
from collections import Counter

class NGramVectorizer:
    
    def __init__(self, ngram_min = 1, ngram_max = 1, min_count = 1):
            #inisialisasi ngram range = ngram_max+1 - ngram min
        self.ngram_range = range(ngram_min, ngram_max + 1)
            #inisialisasi min count yang berfungsi sebagai batasan
        self.min_count = min_count
            #mengubah kalimat menjadi token dan menghitung masing-masing kosakata dalam dokumen untuk membuat vektor tiap kalimat.  
    def fit_transform(self, data):
        dataset_ngram = [] #empty list yang berisi kumpulan hasil dari sentence ngram
        self.vocab_dictionary = list()
        for sentence in data:
            #memisahkan kalimat ke bentuk token
            tokens = sentence.split(' ')
            sentence_ngram = {} #empty dictionary yang akan berisi kalimat-kalimat ngram
            for ngram_limit in self.ngram_range:
                #buat ambil ngramnya - pertama hitung RANGE dengan mengurangi panjang token dan ngram limit(didapat dari ngram range). 
                #setelah range sudah dihitung, token dari elemen (inclusive) ke elemen (ekslusif)
                #untuk tiap array akan dijoin lagi menjadi list of array
                ngram_result = [' '.join(tokens[i:i+ngram_limit]) for i in range(len(tokens)-ngram_limit+1)] #buat ambil ngramnya
                for ngram_instance in ngram_result:
                    #membuat vektor dengan menghitung dari jumlah masing2 token
                    #jika ngram_instancenya tidak ada di sentence_ngram, nanti token samadengan 1
                    #jika ngram_instance ada di sentence_ngram, token sebelumnya akan ditambah 1
                    if ngram_instance not in sentence_ngram:
                        sentence_ngram[ngram_instance] = 1
                    else:
                        sentence_ngram[ngram_instance] += 1
                    #menambahkan ngram_result ke vocab_dictionary
                self.vocab_dictionary.extend(ngram_result)
                #menambahkan sentence ngram ke dataset_ngram
            dataset_ngram.append(sentence_ngram)
    
        filtered_dictionary = [] #empty list yang berisikan dictionary yang sudah terfilter

        vocab_dict_count = dict(Counter(self.vocab_dictionary))
        for key in vocab_dict_count:
            #jika key nya vocab_dict_count lebih besar dari min_count (batasannya)
            if vocab_dict_count[key] >= self.min_count:
                #maka menambahkan key ke fitered dictionary
                filtered_dictionary.append(key)

        #memastikan tidak ad yg duplikat
        self.vocab_dictionary = set(filtered_dictionary)
        
        #memotong dataset ngram
        dataset_transformed = [dict.fromkeys(self.vocab_dictionary, 0) for i in range(0, len(data))]

        #matrix dibuat berdasarkan panjang dataset_transformednya
        for data_idx in range(0, len(dataset_transformed)):
            #iterasi sebanyak data index dalam dataset_ngramnya
            for ngram in dataset_ngram[data_idx]:
                #kalau ngramnya ada di vocab_dictionary akan terus dibuat matrixnya
                if(ngram in self.vocab_dictionary):
                    dataset_transformed[data_idx][ngram] += 1

        dataset_transformed = pd.DataFrame(dataset_transformed)
        return dataset_transformed
    
    def transform(self,data):
        dataset_ngram = []
        for sentence in data:
            tokens = sentence.split(' ')
            sentence_ngram = {}
            for ngram_limit in self.ngram_range:
                ngram_result = [ ' '.join(tokens[i:i+ngram_limit]) for i in range(len(tokens)-ngram_limit+1)]
                for ngram_instance in ngram_result:
                    if ngram_instance not in sentence_ngram:
                        sentence_ngram[ngram_instance] = 1
                    else:
                        sentence_ngram[ngram_instance] += 1
            dataset_ngram.append(sentence_ngram)

        dataset_transformed = [dict.fromkeys(self.vocab_dictionary, 0) for i in range(0, len(data))]

        for data_idx in range(0, len(dataset_transformed)):
            for ngram in dataset_ngram[data_idx]:
                if(ngram in self.vocab_dictionary):
                    dataset_transformed[data_idx][ngram] += 1

        dataset_transformed = pd.DataFrame(dataset_transformed)
        return dataset_transformed

X_train, X_test, y_train, y_test = train_test_split(corpus, list(dataset.id_sentimen), test_size= 0.20, random_state=0)

ngram_vectorizer = NGramVectorizer(3,3,3)

X_train = ngram_vectorizer.fit_transform(X_train)
X_train.head()

X_test = ngram_vectorizer.transform(X_test)
X_test.head()

import math

# class helper perhitungan TF-IDF
class TFIDFHelper:
  def __init__(self):
    pass

  # fungsi ini digunakan untuk menyimpan informasi terkait dengan term pada data yang akan digunakan.
  # fungsi ini juga digunakan untuk menghitung nilai IDF dari setiap term
  def fit_transform(self,data):
    # berisikan data untuk digunakan ke depannya.
    self.data = data

    # menyimpan seluruh term yang terdapat dalam data yang sudah diproses pada NGramVectorizer
    self.terms = list(data.columns)
    

    # jumlah seluruh dokumen untuk perhitungan nilai IDF
    self.number_of_documents = data.shape[1]

    self.IDF = {} #empty dictionary, nanti beisi nilai idf

    # menghitung nilai IDF dari seluruh term
    for term in self.terms:
      # rumus IDF = log2( (total_dokumen + 1) / (total_kemunculan_term + 1) ) + 1
      self.IDF[term] = math.log2( (self.number_of_documents + 1) / (data[term].sum() + 1) ) + 1
    
    return self.transform(data)

  # fungsi ini digunakan untuk mengubah data n-gram ke bobot TF-IDF
  def transform(self,data):
    result = []
    #ignore data_Idx (anonymous variable)
    for _,row in data.iterrows():
      data_transformed = {} #berisikan data_transformed NGgramVectorizer
      for term in self.terms:
          #mengkalikan row dalam term dengan idf dalam term
        data_transformed[term] = row[term] * self.IDF[term]
      result.append(data_transformed)
    result = pd.DataFrame(result)
    return result


tfidfvectorizer = TFIDFHelper()

X_train = tfidfvectorizer.fit_transform(X_train)
X_test = tfidfvectorizer.transform(X_test)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
ytest = np.array(y_test)

print(classification_report(ytest, y_pred))
print(confusion_matrix(ytest, y_pred))

kalimat_baru = ['apliksinya mantap gan']
kalimat_baru = ngram_vectorizer.transform(kalimat_baru) 

kalimat_baru = tfidfvectorizer.transform(kalimat_baru)
                
classifier.predict(kalimat_baru)

