# -*- coding: utf-8 -*-
import codecs
from sklearn import svm
import numpy as np
import csv
import fastText
from sklearn.externals import joblib


# f = open('train_tweets.txt',encoding='utf-8')
# f_labeled = open('new_train_tweets.txt','w',encoding='UTF-8')
# for line2 in f:
#     arr = line2.split("\t")
#     newText = "__label__" + arr[0] +" "+ arr[1]
#     f_labeled.write(newText+"\n")
# f_labeled.close()

# target = np.array(targetArray)
# allrignt=0;
# all=0
# f2 = open('../2019S1-KTproj2-data/train-tweets.txt','r', encoding='UTF-8')
# f3 = open("newfile.txt",'w',encoding='UTF-8')
# for (line1, line2) in zip(f2, target) :
#     if ("love"in line1 or "happy" in line1 or "best" in line1 or "fuck" in line1 or "great" in line1 or "not" in line1
#      or "trump" in line1 or "my" in line1):
#         allrignt+=1;
#     all+=1
#     newT = "__label__"+line2
#     newL = (np.array(line1.split())[1:]).tolist()
#     # print()
#     f3.write(newT+" "+" ".join(newL)+"\n")
# f3.close()
# print ("all:",all," contains:",allrignt)
#
model = fastText.train_supervised(
        input="new_train_tweets.txt", epoch=25, lr=1.0, wordNgrams=2, verbose=2, minCount=1
    )
model.save_model("sml_model.ftz")

print("done")

