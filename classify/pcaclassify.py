import random
import re

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
from sklearn.decomposition import PCA, TruncatedSVD
from zipfile import ZipFile
from urllib.request import urlopen
file = open('../data/ntrain.txt','r',encoding='utf-8')
allContents = file.readlines()

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
        if(len(cases)<2):
            print(line)
            continue
        texts.append(cases[1].strip())
        labels.append(cases[0].strip())
    tfidffeatures = countVectorizer.transform(texts)
    # pca = PCA(n_components=10000)
    # newtffeature = pca.fit_transform(tfidffeatures)
    # print(pca.explained_variance_ratio_)
    return tfidffeatures, labels



bowtraintext=[]
ngramtext=[]
countVectorizer = CountVectorizer(ngram_range=(1, 2),dtype=np.int32)
analyze = countVectorizer.build_analyzer()
testind = 0
newAllContents=[]
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
    newAllContents.append(cc[0]+"\t"+cc[1])
    bowtraintext.append(cc[1].strip())
vectorizer = TfidfVectorizer()
vectorizer.fit(bowtraintext)
countVectorizer.fit(bowtraintext)
allfeatures, allLabels = getFeaturesAndLabelBow(newAllContents,vectorizer,countVectorizer)
print(allfeatures.shape)
# tsvd = TruncatedSVD(n_components=10000)
# allfeatures = tsvd.fit_transform(allfeatures)
# print(tsvd.explained_variance_ratio_)

trainFeatures = allfeatures[:splitPos]
testFeatures = allfeatures[splitPos:]
trainLabels = allLabels[:splitPos]
testLabels = allLabels[splitPos:]
# print(len(countVectorizer.vocabulary))
# trainFeatures, trainLabels = getFeaturesAndLabelBow(train,vectorizer,countVectorizer)
# testFeatures, testLabels = getFeaturesAndLabelBow(test,vectorizer,countVectorizer)
# trainFeatures, trainLabels = getFeaturesAndLabel(train)
# testFeatures, testLabels = getFeaturesAndLabel(test)
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
# joblib.dump(tsvd,"pca.mmm")
