# train_tweets.txt 共328932条 80% 为263146 20%为65786
import random
import re
# 第一次分割
file = open('train_tweets.txt','r',encoding='utf-8')
allContents = file.readlines()

random.shuffle(allContents)
stopwords = ["a","A","The","the"]
newContents=[]
for line in allContents:
    case = line.split("\t")
    words = case[1].split(" ")
    newWords = [each for each in words if each not in  stopwords]

    nl = ' '.join(newWords)
    print(nl)
    newContents.append(case[0] + "\t" + nl.lower())


trainDataSet = newContents
testDataSet = newContents[263146:]

newTrain = open('trainDataSetALLShuff','w',encoding='utf-8')
newTrain.writelines(trainDataSet)

newTest = open('testDataSet','w',encoding='utf-8')
newTest.writelines(testDataSet)

newTrain.close()
newTest.close()
file.close()