import random

import gensim
import numpy as np
from sklearn.externals import joblib
file = open('../data/train_tweets.txt','r',encoding='utf-8')
allContents = file.readlines()
wordVec = gensim.models.KeyedVectors.load_word2vec_format("../textrcnn/word2Vec.bin", binary=True)
random.shuffle(allContents)
splitPos = int(len(allContents)*0.8)
train = allContents[:splitPos]
test = allContents[splitPos:]

stopwords = ["a","A","The","the"]
newContents=[]

def getFeaturesAndLabel(lines):
    labels = []
    wordfeatures = []
    for line in lines:
        case = line.split("\t")
        words = case[1].strip().split()
        val = np.zeros(200)
        we = []
        num = 0
        for word in words:
                try:
                    vector = wordVec.wv[word]
                    val += vector
                    num+=1
                except Exception as e:
                    print(word + "不存在于词向量中")
        # print(num)
        we = val/num
        wordfeatures.append(we)
        # print(we)
        labels.append(case[0].strip())
    return wordfeatures, labels

trainFeatures, trainLabels = getFeaturesAndLabel(train)
testFeatures, testLabels = getFeaturesAndLabel(test)

print("done")
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(trainFeatures, trainLabels)
prelabels = model.predict(testFeatures)
right = 0.0
all = 0.0
for i in  range(len(prelabels)):
    all+=1
    if prelabels[i] == testLabels[i]:
        right+=1

print(right/all)
joblib.dump(model,"logistic_model.m")
