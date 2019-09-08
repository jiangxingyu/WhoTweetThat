import random
import re
import os
import gensim
import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from zipfile import ZipFile
from urllib.request import urlopen
file = open('../data/train_tweets.txt','r',encoding='utf-8')
allContents = file.readlines()
# wordVec = gensim.models.KeyedVectors.load_word2vec_format("../textrcnn/word2Vec.bin", binary=True)
embeddings_file = '../glove.6B/glove.6B.300d.txt'.format(300)
word2vec_output_file = '{0}.word2vec'.format(embeddings_file)
glove2word2vec(embeddings_file, word2vec_output_file)
glove = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
random.shuffle(allContents)
splitPos = int(len(allContents)*0.99)
train = allContents[:splitPos]
test = allContents[splitPos:]

stopwords = ["a","A","The","the"]
newContents=[]
labels = []
wordfeatures = []

def getFeaturesAndLabelBow(lines,vectorizer,countVectorizer):
    analyze = countVectorizer.build_analyzer()
    labels=[]
    texts=[]
    for line in lines:
        line = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''','@HTTP',line)
        print(line)
        cases = line.split("\t")
        # tokens = list(analyze(cases[1]))
        # newtokens=[]
        # for t in tokens:
        #     nt=re.sub("\s","",t)
        #     newtokens.append(nt)
        # texts.append(" ".join(newtokens).strip())
        texts.append(cases[1].strip())
        labels.append(cases[0].strip())
    tfidffeatures = countVectorizer.transform(texts)
    # pca = PCA(n_components=10000)
    # newtffeature = pca.fit_transform(tfidffeatures)
    # print(pca.explained_variance_ratio_)
    return tfidffeatures, labels

def getFeaturesAndLabel(lines):
    min = 0
    for line in lines:
        HTTPNum = 0
        stopwordsNums=0
        specialfeatureNums=0

        line = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''','@HTTP',line)
        case = line.split("\t")
        words = case[1].strip().split()
        val = np.zeros(300)
        we = []
        num = 0
        for word in words:
                for i in word:
                    if i < 'A' or i > 'z':
                        specialfeatureNums+=1
                if re.search("@HTTP", word, flags=0):
                    HTTPNum+=1

                if word in stopwords:
                    stopwordsNums+=1
                try:
                    vector = embeddings_index[word]
                    val += vector
                    num+=1
                except Exception as e:
                    print(e)
                    print(word + "不存在于词向量中")
        if num == 0:
            continue
        we = val/num
        nwe = np.array(list(we)+[stopwordsNums, specialfeatureNums, HTTPNum])
        mmm = nwe.min()
        if min > mmm:
            min = mmm;

        wordfeatures.append(nwe)
        # print(we)
        labels.append(case[0].strip())
    for i in range(len(wordfeatures)):
        wordfeatures[i] -= min
    return wordfeatures, labels

bowtraintext=[]
ngramtext=[]
countVectorizer = CountVectorizer(ngram_range=(1, 2),dtype=np.int32)
analyze = countVectorizer.build_analyzer()
testind = 0
for line in allContents:
    print(testind)
    testind+=1
    line = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''','@HTTP',line)
    cc = line.split("\t")
    # tokens = list(analyze(cc[1].strip()))
    # newtokens=[]
    # for t in tokens:
    #     nt=re.sub("\s","",t)
    #     newtokens.append(nt)
    # nline = " ".join(newtokens).strip()
    bowtraintext.append(cc[1].strip())
vectorizer = TfidfVectorizer()
vectorizer.fit(bowtraintext)
countVectorizer.fit(bowtraintext)
# print(len(countVectorizer.vocabulary))
# trainFeatures, trainLabels = getFeaturesAndLabelBow(train,vectorizer,countVectorizer)
# testFeatures, testLabels = getFeaturesAndLabelBow(test,vectorizer,countVectorizer)
trainFeatures, trainLabels = getFeaturesAndLabel(train)
testFeatures, testLabels = getFeaturesAndLabel(test)
print(len(set(trainLabels)))

print("done")
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, SGDClassifier

# model = LogisticRegression()
model = SGDClassifier(loss="log")
# model = MultinomialNB()
model.fit(trainFeatures, trainLabels)
prelabels = model.predict(testFeatures)
right = 0.0
all = 0.0
for i in range(len(prelabels)):
    all+=1
    if prelabels[i] == testLabels[i]:
        right+=1

print(right/all)
joblib.dump(model,"lgtest.m")
