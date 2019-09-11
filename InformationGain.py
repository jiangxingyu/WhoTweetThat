# 总推数 328932
# 含emoji推数12204

emojiList = [':-\\)',';\\(',';\\)','\*_\*','\^_\^','-_-','-\.-',':\\(',':\\)']
stopWordsList = ['a', 'the', 'of', 'for', 'to', 'at', 'in', 'on', 'and']

import re
from sklearn import tree


file = open("data/train_tweets.txt", "r", encoding="utf-8")
allContents = file.readlines()
# counter = 0
# for emoji in emojiList:
#     for each in allContents:
#         if re.search(emoji,each):
#             counter+=1
#
# print('含emoji推数' + str(counter))

# labelList = []
# dict = {}
# for each in allContents:
#     cells = each.split("\t")
#     val = dict.get(cells[0],0)
#     dict[cells[0]] = val+1
#
# print("每个作者推数"+str(dict))
#
# newDict = {}
# for each in allContents:
#     cells = each.split("\t")
#     val = newDict.get(cells[0],0)
#     counter = 0
#     for emoji in emojiList:
#         if re.search(emoji,each):
#             counter+=1
#     newDict[cells[0]] = val + counter
#
# print("每个作者含emoji的推数"+str(newDict))

labelList = []
featureList = []
emojiCounter = 0
lowerCount = 0
upperCount = 0

for each in allContents:
    tempList = []
    cells = each.split('\t')
    labelList.append(cells[0])

    for emoji in emojiList:
        emojiResult = re.findall(emoji,each)
        emojiCounter = len(emojiResult)
    tempList.append(emojiCounter)

    lowerCount = sum(map(str.islower, each))
    tempList.append(lowerCount)

    upperCount = sum(map(str.isupper, each))
    tempList.append(upperCount)

    symbolResult = re.findall('\W', each)
    sR = len(symbolResult)
    tempList.append(sR)

    stCounter = 0
    each = each.lower()
    n1 = each.split('\t')
    n2 = n1[1].split(' ')
    for j in n2:
        if j in stopWordsList:
            stCounter+=1

    tempList.append(stCounter)
    featureList.append(tempList)

print(len(labelList))
print(len(featureList))
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit( featureList,labelList)
print(clf.feature_importances_)


