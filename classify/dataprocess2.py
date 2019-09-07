import random
import re
import os
import gensim
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from zipfile import ZipFile
from urllib.request import urlopen
file = open('../data/1kdata.txt','r',encoding='utf-8')
allContents = file.readlines()
wordVec = gensim.models.KeyedVectors.load_word2vec_format("../textrcnn/word2Vec.bin", binary=True)
embeddings_file = '../glove.6B'
embeddings_index={}
f = open(os.path.join(embeddings_file, 'glove.6B.100d.txt'),'r',encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
word2vec_output_file = '{0}.word2vec'.format(embeddings_file)
glove2word2vec(embeddings_file, word2vec_output_file)
glove = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
random.shuffle(allContents)
splitPos = int(len(allContents)*0.9)
train = allContents[:splitPos]
test = allContents[splitPos:]

stopwords = ["a","A","The","the"]
newContents=[]
labels = []
wordfeatures = []

def getFeaturesAndLabelBow(lines,vectorizer):
    labels=[]
    texts=[]
    for line in lines:
        print(line)
        cases = line.split("\t")

        texts.append(cases[1])
        labels.append(cases[0])
    features = vectorizer.transform(texts)

    return features, labels

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
for line in allContents:
    line = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''','@HTTP',line)
    cc = line.split("\t")
    bowtraintext.append(cc[1])
vectorizer = TfidfVectorizer()
vectorizer.fit(bowtraintext)

trainFeatures, trainLabels = getFeaturesAndLabelBow(train,vectorizer)
testFeatures, testLabels = getFeaturesAndLabelBow(test,vectorizer)
# trainFeatures, trainLabels = getFeaturesAndLabel(train)
# testFeatures, testLabels = getFeaturesAndLabel(test)
print(len(set(trainLabels)))

print("done")
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver="sag",multi_class="ovr")
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
# joblib.dump(model,"lgtest.m")
