# train_tweets.txt 共328932条 80% 为263146 20%为65786
import random
# 第一次分割
# file = open('train_tweets.txt','r')
# allContents = file.readlines()
#
# random.shuffle(allContents)
#
# for line in allContents:
#     print(line)
#
# trainDataSet = allContents[:263146]
# testDataSet = allContents[263146:]
#
# newTrain = open('trainDataSet','a')
# newTrain.writelines(trainDataSet)
#
# newTest = open('testDataSet','a')
# newTest.writelines(testDataSet)
#
# newTrain.close()
# newTest.close()
# file.close()

###############################################################
###############################################################
###############################################################
# 第二次分割

# # remove stopwords for test data set
# # stopWordsSet = ['a','the','of','The','A','an','An']
#
# file = open('testDataSet','r')
# allcontents = file.readlines()
#
# cleanedList = []
# for i in range(len(allcontents)):
#     newLine1 = allcontents[i].split(' ',)
#     newLine2 = allcontents[i].split('\t',)
#
#
#     newLine3 = newLine2[1].split(' ')
#     #print(newLine3)
#     newLine4 = [each for each in newLine3 if each != 'a']
#     newLine5 = [each for each in newLine4 if each != 'the']
#     newLine6 = [each for each in newLine5 if each != 'of']
#     newLine7 = [each for each in newLine6 if each != 'The']
#     newLine8 = [each for each in newLine7 if each != 'A']
#     newLine9 = [each for each in newLine8 if each != 'an']
#     newLine10 = [each for each in newLine9 if each != 'An']
#     #print(newLine10)
#     tempStr = ' '.join(newLine10)
#     finalStr = newLine2[0] + ' ' + tempStr
#     print(finalStr)
#     cleanedList.append(finalStr)
#
# newTest = open('testDataSet2','a')
# newTest.writelines(cleanedList)
# newTest.close()

###############################################################
# remove stopwords for train data set
# stopWordsSet = ['a','the','of','The','A','an','An']

# file = open('trainDataSet','r')
# allcontents = file.readlines()
#
# cleanedList = []
# for i in range(len(allcontents)):
#     newLine1 = allcontents[i].split(' ',)
#     newLine2 = allcontents[i].split('\t',)
#
#
#     newLine3 = newLine2[1].split(' ')
#     #print(newLine3)
#     newLine4 = [each for each in newLine3 if each != 'a']
#     newLine5 = [each for each in newLine4 if each != 'the']
#     newLine6 = [each for each in newLine5 if each != 'of']
#     newLine7 = [each for each in newLine6 if each != 'The']
#     newLine8 = [each for each in newLine7 if each != 'A']
#     newLine9 = [each for each in newLine8 if each != 'an']
#     newLine10 = [each for each in newLine9 if each != 'An']
#     #print(newLine10)
#     tempStr = ' '.join(newLine10)
#     finalStr = newLine2[0] + ' ' + tempStr
#     print(finalStr)
#     cleanedList.append(finalStr)
#
# newTest = open('trainDataSet2','w')
# newTest.writelines(cleanedList)
# newTest.close()

###############################################################
###############################################################
###############################################################
