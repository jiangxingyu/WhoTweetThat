# train_tweets.txt 共328932条 80% 为263146 20%为65786
import random
# 第一次分割
file = open('train_tweets.txt','r',encoding='utf-8')
allContents = file.readlines()

random.shuffle(allContents)

for line in allContents:
    print(line)

trainDataSet = allContents[:263146]
testDataSet = allContents[263146:]

newTrain = open('trainDataSet','a',encoding='utf-8')
newTrain.writelines(trainDataSet)

newTest = open('testDataSet','a',encoding='utf-8')
newTest.writelines(testDataSet)

newTrain.close()
newTest.close()
file.close()