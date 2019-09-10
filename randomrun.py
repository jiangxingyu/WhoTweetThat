# data process
import random
import re
import fasttext

from nltk import word_tokenize

f = open('data/1kdata.txt',encoding='utf-8')

lines = f.readlines()
pos = int(0.6*len(lines))
i=0
for i in range(15):
    f_labeled = open('new_train_tweets_all.txt', 'w', encoding='UTF-8')
    f_labeled2 = open('test_new.txt', 'w', encoding='UTF-8')
    random.shuffle(lines)
    temLines = lines[:pos]
    for line2 in lines:
        line2 = re.sub(" +", " ", line2)
        arr = line2.split("\t")
        tokens = word_tokenize(arr[1])
        arr[1] = " ".join([i for i in tokens if i not in ['A','a','The','THE','the']])

        newText = "__label__" + arr[0] + " " + arr[1]
        if i < pos:
            f_labeled.write(newText + "\n")
        else:
            f_labeled2.write(newText + "\n")

    f_labeled.close()
    f_labeled2.close()
    model = None
    if i%2 == 0:
        model = fasttext.train_supervised(
            input="new_train_tweets_all.txt", wordNgrams=2, ws=5, verbose=2, thread=7, t=0.0001, neg=5, minn=3,
            minCountLabel=0, minCount=1, maxn=6, lrUpdateRate=100, epoch=40, lr=2.2389885442286235, bucket=1084286, dim=179
        )
    else:
        model = fasttext.train_supervised(
            input="new_train_tweets_all.txt", wordNgrams=2, ws=5, verbose=2, thread=7, t=0.0001, neg=5, minn=2,
            minCountLabel=0, minCount=1, maxn=5, lrUpdateRate=100, epoch=40, lr=2.7185326849317737, bucket=2534640,
            dim=107
        )
    model.save_model("model/fasttext_"+str(i)+".ftz")
    if i>5:
        pos=0.4