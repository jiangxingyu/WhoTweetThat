# -*- coding: utf-8 -*-
import codecs
import random

from sklearn import svm
import numpy as np
import csv
import fasttext
from sklearn.externals import joblib


f = open('data/train_tweets.txt',encoding='utf-8')
f_labeled = open('new_train_tweets_all.txt','w',encoding='UTF-8')
f_labeled2 = open('test_new.txt','w',encoding='UTF-8')
lines = f.readlines()
random.shuffle(lines)
pos = int(1*len(lines))
i=0

for line2 in lines:
    arr = line2.split("\t")
    newText = "__label__" + arr[0] +" "+ arr[1]
    if i< pos:
        f_labeled.write(newText.lower()+"\n")
    else:
        f_labeled2.write(newText.lower()+"\n")
    i+=1
print(i)
f_labeled.close()
f_labeled2.close()
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
model = fasttext.train_supervised(
        input="new_train_tweets_all.txt", wordNgrams=2, ws=5,verbose=2,thread=7,t=0.0001,neg=5,minn=2,minCountLabel=0,minCount=1,maxn=5,lrUpdateRate=100,epoch=30, lr=2.718532684931737, bucket=2534640,dim=107
    )

# model = fasttext.train_supervised(
#         epoch=30,input="new_train_tweets_all.txt",autotuneValidationFile="test_new.txt", autotuneMetric="f1",autotuneDuration=1800
#     )
a = model.f.getArgs()

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))
# a = model.f.getArgs()

print("test")
model.save_model("nft2.ftz")

# print_results(*model.test('test_new.txt'))
# model.save_model("sml_model_all_v.ftz")

print("done")

