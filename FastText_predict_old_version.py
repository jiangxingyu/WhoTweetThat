# -*- coding: utf-8 -*-
import numpy as np
import csv
import fasttext
from sklearn.metrics import classification_report


model = fasttext.load_model("sml_model_o.ftz")

# su = 0.0
# right = 0.0
#
# target = []
# predict = []
# for (line,line2) in zip(f,ft):
#
#     su = su+1
#     newL = (np.array(line.split())[1:]).tolist()
#     print(model.predict(" ".join(newL))[0][0])
#     predict.append(model.predict(" ".join(newL))[0][0])
#     target.append("__label__"+(line2.split())[1])
#     if model.predict(" ".join(newL))[0][0] == ("__label__"+(line2.split())[1]).strip() :
#         right = right+1;
#     fn.write("__label__"+(line2.split())[1]+" "+" ".join(newL)+"\n")
#
# print(classification_report(target, predict))
# print ("sum:",su," ;right:",right)
# print ("accuracy:",right/su)
# fn.close()
def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))
a = model.f.getArgs()

print("test")
print_results(*model.test('new_train_tweets_all.txt'))
# print_results(*model.test('new_train_tweets.txt'))

