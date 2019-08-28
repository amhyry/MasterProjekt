# coding: utf8
#import warnings
#warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
#our own classes
import JsonToSentencesConverter as Crawler

#basics
import pprint
import sys
import locale
import io
import multiprocessing
import re
import string
import os
from datetime import datetime
import numpy as np
import pandas as pd


import json
import re
#new import
from collections import Counter

#gensim
from gensim.models import TfidfModel
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.summarization import keywords
from gensim import corpora
from gensim.models.ldamodel import LdaModel
#new import
from gensim.models.ldamulticore import LdaMulticore

import nltk
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.manifold import TSNE
# hyperparameter training imports
from sklearn.model_selection import GridSearchCV
#new import
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

#plotting
import matplotlib.pyplot as plt
import seaborn as sns

# visualization imports
from IPython.display import display
import matplotlib.image as mpimg
import base64
import io

#polyglot


#%matplotlib inline
sns.set()  # defines the style of the plots to be seaborn style




#correct the individual encoding
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')
pd.set_option('display.max_columns', None)

class NLP_NeuralNet:
    """Class docstrings go here.
    ...

    Args:
        json_file : str
            filename/filepath to jsonfile, which contains sentences
        vocabulary_size : int
            [optional] Size of Vocabulary

    Methods
        _initialize(sentences_list=None):
            Prints the animals name and what sound it makes
    """

    def __init__(self, test = True):
        self._initialized = False
        print('Start')
        #self.source = ""
        #self.destination = ""
        #self.dir_up = ""
        self.smalldata = test
        self.modelstr = "_33001"

        if(os.name == "posix"):
            self.dir_up = '..'
        if(os.name == "nt"):
            self.dir_up = '.'
#        print("Path: " + os.name + dir_up)
        #version = sum([int(item) for item in datetime.now().strftime('%m_%d_%H_%M').split('_')])
        version = datetime.now().strftime('%m_%d_%H_%M')
        if self.smalldata == True:
            self.modelstr = "_1001_" + version
            self.source = self.dir_up + '/Data/sampleFromDataCrowlerindeed1001.json'
            self.destination =  self.dir_up + '/Data/data_for_voc_1001.json'
        else:
            self.modelstr = "_33001_" + version
            self.source = self.dir_up + '/Data/sampleFromDataCrowlerindeed33001.json'
            self.destination = self.dir_up + '/Data/data_for_voc_33001.json'

    def convert(self):
        Crawler.converter(self.source, self.destination)
        print(self.source + " konvertiert in " + self.destination)

    def loadData(self):
        with open(self.destination, encoding='utf-8') as json_file:
            self.data = json.load(json_file)
        self.DF_data = pd.read_json(self.destination, encoding='utf-8')
        self.DF_data.rename(columns={0:'text'},inplace=True)
        print(self.destination + ' Daten geladen in data und DF_data')
        return self.data, self.DF_data

    def makeBOW(self, data, load=None):
        if load == None:
            print("TODO:: vielleicht aus Vocabulary laden?")
        else:
            load = self.dir_up + '/Data/' + load
            with open(load, encoding='utf-8') as json_file:
                bow = json.load(json_file)

        print(load + ' Bag of words geladen')
        self.bow = bow
        return bow

    def makeLDA(self, data, load=None):
        if load == None:
            print("LDA Model wird erzeugt... Daten vorbereiten.")
            doc_clean = [clean(doc).split() for doc in data]
            print("Model wird erzeugt... Dictionary wird angelegt.")
            dictionary = corpora.Dictionary(doc_clean)
            print("Model wird erzeugt... Dictionary angelegt. Erzeuge Matrix...")
            #print(dictionary)
            doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
            print("Model wird erzeugt... Matrix erstellt. Beginne Training.")
            #print(doc_term_matrix)
            ldamodel = LdaModel(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)
            print("LDA Model wurde erzeugt. Model abspeichern...")
            ldamodel.save(self.dir_up + '/Data/Model/lda_model'+self.modelstr)
            print("LDA Model erstellt unter: "+ self.dir_up + '/Data/Model/lda_model'+self.modelstr)
        else:
            load = self.dir_up + '/Data/Model/' + load
            ldamodel = LdaModel.load(load)
            #ldamodel = LdaModel.load(dir_up + '/Data/Model/ldamodel')
        #print(ldamodel.print_topics(num_topics=3, num_words=10))
        self.lda = ldamodel
        print(load + " ldamodel  erfolgreich geladen!")
        return ldamodel

    def makeW2V(self, data, load=None):
        if load == None:
            print("W2V Model wird erzeugt... Daten vorbereiten.")
            EMB_DIM = 300
            data_clean = [doc.split() for doc in data]
            print("Model wird erzeugt... Daten vorbereitet. Beginne training...")

            w2v = Word2Vec(data_clean, size=EMB_DIM, window=5, min_count=5, negative=15, iter=10, workers=multiprocessing.cpu_count())
            print("W2V Model wurde erzeugt. Model abspeichern...")
            w2v.save(self.dir_up+'/Data/Model/wtov_model'+self.modelstr)
            print("W2V Model erstellt unter: "+ self.dir_up+'/Data/Model/wtov_model'+self.modelstr)
        else:
            load = self.dir_up + '/Data/Model/' + load
            #ldamodel = LdaModel.load(load)
            #w2v = Word2Vec.load(dir_up+'/Data/Model/wtov_model'+modelstr)
            w2v = Word2Vec.load(load)
        self.w2v = w2v
        print(load + " w2vmodel erfolgreich geladen!")
        return w2v
        #wv = w2v.wv



#end of class-------------------------------------------------------------------
stop = set(stopwords.words('german'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

def removeEmptyListsFromDocs(docs):
    result = []
    for doc in docs:
        if isinstance(doc,str):
            result.append(doc)
    return result


def clean(doc):
    stop_free = " ".join([i for i in doc.split() if i.lower() not in stop])
    punct_free = ''.join([ch for ch in stop_free if ch not in exclude])
    normalized = " ".join([lemma.lemmatize(word) for word in punct_free.split()])
    return normalized

def umlauteConverter(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.encode("utf-8",'replace')
    text = text.decode('utf-8','replace')
    return text

def sentence_nomalizer(text):
    result = []
    text = umlauteConverter(text)
    #text = [w for w in text.split() ]
    #for value in text:
    #    if len(value) >= 1: result.append(value)

    return text

def dict2listConverter(dataAsDict):
    result = []
    for key in dataAsDict.keys():
        result.append(dataAsDict[key])
    return result


#NEUE FUNKTIONS, für plotting---------------------------------------------------
def remove_ascii_words(df):
    """ removes non-ascii characters from the 'texts' column in df.
    It returns the words containig non-ascii characers.
    """
    our_special_word = 'qwerty'
    non_ascii_words = []
    for i in range(len(df)):
        for word in df.loc[i, 'text'].split(' '):
            if any([ord(character) >= 128 for character in word]):
                non_ascii_words.append(word)
                df.loc[i, 'text'] = df.loc[i, 'text'].replace(word, our_special_word)
    return non_ascii_words

def w2v_preprocessing(df):
    """ All the preprocessing steps for word2vec are done in this function.
    All mutations are done on the dataframe itself. So this function returns
    nothing.
    """
    df['text'] = df.text.str.lower()
    df['document_sentences'] = df.text.str.split('.')  # split texts into individual sentences
    df['tokenized_sentences'] = list(map(lambda sentences:
                                         list(map(nltk.word_tokenize, sentences)),
                                         df.document_sentences))  # tokenize sentences
    df['tokenized_sentences'] = list(map(lambda sentences:
                                         list(map(get_good_tokens, sentences)),
                                         df.tokenized_sentences))  # remove unwanted characters
    df['tokenized_sentences'] = list(map(lambda sentences:
                                         list(filter(lambda lst: lst, sentences)),
                                         df.tokenized_sentences))  # remove empty lists

def lda_get_good_tokens(df):
    df['text'] = df.text.str.lower()
    df['tokenized_text'] = list(map(nltk.word_tokenize, df.text))
    df['tokenized_text'] = list(map(get_good_tokens, df.tokenized_text))

def get_good_tokens(sentence):
    replaced_punctation = list(map(lambda token: re.sub('[^0-9A-Za-zßÄÖÜöäü!?]+', '', token), sentence))
    removed_punctation = list(filter(lambda token: token, replaced_punctation))
    return removed_punctation

def stem_words(df):
    lemm = nltk.stem.WordNetLemmatizer()
    df['lemmatized_text'] = list(map(lambda sentence:
                                     list(map(lemm.lemmatize, sentence)),
                                     df.stopwords_removed))

    p_stemmer = nltk.stem.porter.PorterStemmer()
    df['stemmed_text'] = list(map(lambda sentence:
                                  list(map(p_stemmer.stem, sentence)),
                                  df.lemmatized_text))

def remove_stopwords(df):
    """ Removes stopwords based on a known set of stopwords
    available in the nltk package. In addition, we include our
    made up word in here.
    """
    our_special_word = 'qwerty'
    # Luckily nltk already has a set of stopwords that we can remove from the texts.
    stopwords = nltk.corpus.stopwords.words('german')
    # we'll add our own special word in here 'qwerty'
    stopwords.append(our_special_word)

    df['stopwords_removed'] = list(map(lambda doc:
                                       [word for word in doc if word not in stopwords],
                                       df['tokenized_text']))
def document_to_bow(df):
    df['bow'] = list(map(lambda doc: dictionary.doc2bow(doc), df.stemmed_text))

def word_frequency_barplot(df, nr_top_words=50):
    """ df should have a column named count.
    """
    fig, ax = plt.subplots(1,1,figsize=(20,5))

    sns.barplot(list(range(nr_top_words)), df['count'].values[:nr_top_words], palette='hls', ax=ax)

    ax.set_xticks(list(range(nr_top_words)))
    ax.set_xticklabels(df.index[:nr_top_words], fontsize=14, rotation=90)
    return ax

def word_frequency_barplot_pretty(df, nr_top_words=50):
    """ df should have a column named count.
    """
    fig, ax = plt.subplots(1,1,figsize=(20,5))

    #sns.barplot(range(len(df)), list(df.values()), tick_label=list(df.keys()))
    #sns.barplot(list(range(nr_top_words)), df['count'].values[:nr_top_words], palette='hls', ax=ax)
    sns.barplot(list(range(nr_top_words)), list(df.values())[:50], palette='hls', ax=ax)

    ax.set_xticks(list(range(nr_top_words)))

    ax.set_xticklabels(list(sorted_bow.keys())[:50], fontsize=14, rotation=90)
    return ax

#end----------------------------------------------------------------------------

if __name__ == '__main__':
    my_NLP_nn = NLP_NeuralNet(test = True) # test auf false und es werden 33001 daten geladen
    #my_NLP_nn.convert() #Wenn neue Daten benötigt werden.

    data, DF_data = my_NLP_nn.loadData()
    #lda = x.makeLDA(data, load='lda_model_33001')
    #w2v = x.makeW2V(data, load='wtov_model_1001')
    bow = my_NLP_nn.makeBOW(data, load='bag_of_words.vocab') #
    lda = my_NLP_nn.makeLDA(data, load='lda_model_1001_08_25_15_13') #
    w2v = my_NLP_nn.makeW2V(data, load='wtov_model_1001_08_25_15_13') #


    #lda = x.makeLDA(data)
    #w2v = x.makeW2V(data)
###Whatever comes after here, to test and etc.
#plot frequencies of words

    lda_get_good_tokens(DF_data)
    #non_ascii_words = remove_ascii_words(DF_data)
    #print(non_ascii_words)
    #print("Replaced {} words with characters with an ordinal >= 128 in the train data.".format(len(non_ascii_words)))
    w2v_preprocessing(DF_data)
    #lda_get_good_tokens(DF_data)
    remove_stopwords(DF_data)
    #stem_words(DF_data)

    print(DF_data.head(5))
    print(type(DF_data.tokenized_text.values))
    print(DF_data.tokenized_text.values)
    tokenized_only_dict = Counter(np.concatenate(DF_data.stopwords_removed.values))
    #print(type(tokenized_only_dict))
    #print(tokenized_only_dict)
    #document_to_bow(DF_data)
    tokenized_only_df = pd.DataFrame.from_dict(tokenized_only_dict, orient='index')
    tokenized_only_df.rename(columns={0: 'count'}, inplace=True)
    tokenized_only_df.sort_values('count', ascending=False, inplace=True)

    ax = word_frequency_barplot(tokenized_only_df)
    ax.set_title("Word Frequencies", fontsize=16);
    plt.show()

#end plot frequencies of words

    wv = w2v.wv

    print(wv.vocab)
    sorted_bow = sorted(bow.items(), key=lambda x: x[1], reverse=True)[:100]

    print(type(sorted_bow))
    print(sorted_bow)
    sorted_bow = dict(sorted_bow)

    #df = pd.DataFrame(bow.values(), index=bow.keys(), columns=['x', 'y'])
    #df.head(10)

    ax = word_frequency_barplot_pretty(sorted_bow)
    #ax = word_frequency_barplot(tokenized_only_df)
    ax.set_title("Word Frequencies", fontsize=16);
    plt.show()

    #plt.bar(range(len(sorted_bow)), list(sorted_bow.values()), tick_label=list(sorted_bow.keys()))
    #plt.show()

    zwischenergebnis = {}
    topics = ['Sprachen', 'Fähigkeiten', 'Kenntnisse', 'Wissen', 'Profil']
    for topic in topics:
        zwischenergebnis[topic] = wv.similar_by_word(topic)[:30]
    print(zwischenergebnis)

    vocab = list(wv.vocab)
    print("Ende")
    #vocab2 = [word[0] for word in nltk.pos_tag(nltk.word_tokenize(vocab)) if "NN" in word[1]]
    #print(vocab2)
    #X = w2v[vocab]
    #tsne = TSNE(n_components=2)
    #X_tsne = tsne.fit_transform(X)
    #df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1)

    #ax.scatter(df['x'], df['y'])
    #for word, pos in df.iterrows():
    #    ax.annotate(word, pos)
    #plt.show()


    #kwords = []
    #i = 0
    #j = 0
    #for key in data:
    #    i+=1
    #    print(key)
    #    print(data[key])
    #    if isinstance(data[key],list) :
    #        j +=1
    #        continue
    #
    #    #print(type(data[key]))
    #    #print(keywords("".join(data[key])))
    #    text = sentence_nomalizer(data[key])
    #    print(text)
    #    print(keywords(text))
    #    kwords.append(keywords(text))

    #EMB_DIM = 300
    #data_clean = [doc.split() for doc in data]
    #print(data_clean[1])
    #print(multiprocessing.cpu_count())
    #w2v = Word2Vec(data_clean, size=EMB_DIM, window=5, min_count=5, negative=15, iter=10, workers=multiprocessing.cpu_count())
    #w2v.save(dir_up+'/Data/Model/wtov_model'+modelstr)
    #w2v = Word2Vec.load(dir_up+'/Data/Model/wtov_model'+modelstr)
    #wv = w2v.wv

    #print(wv.vocab)

    #with open('./Data/keywords.json', 'w', encoding='utf8') as json_file:
    #    data = json.dumps(kwords, ensure_ascii=False)
    #    json_file.write(str(data))

    #result = wv.similar_by_word('Sprachen')
    #print("Most Similar to 'Sprachen':\n", result[:20])

    #downloader.list(show_packages=False)

    #print(downloader.supported_tasks(lang="en"))
    #print(downloader.supported_languages_table(task="ner2"))

    #value = downloader.download("morph2.fy")
    #value2 = downloader.download("morph2.en")
    #print("Download value: ", value)

    #for sentence in sentences[:100]:
    #    if Text(sentence).language.code == "de":
    #        print(Text(sentence).entities)

    #for sentence in sentences[:100]:
    #    for word in Text(sentence).words:
            #if Text(sentence).language.code == "de":
    #        print(word, " -> ", word.morphemes)

    #for sentence in sentences:
    #    if Text(sentence).language.code == "de":
    #        print("{:<16}{}".format("Word", "POS Tag")+"\n"+"-"*30)
    #        for word, tag in Text(sentence).pos_tags:
    #            print(u"{:<16}{:>2}".format(word,tag))

    #print("{:<16}{}".format("Word", "POS Tag")+"\n"+"-"*30)
    #for word, tag in text.pos_tags:
    #    print(u"{:<16}{:>2}".format(word, tag))
