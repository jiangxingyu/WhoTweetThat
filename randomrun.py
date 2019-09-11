# data process
import csv
import random
import re
import fasttext

from nltk import word_tokenize

f_test = open('data/test_tweets_unlabeled.txt',encoding='utf-8')

f = open('data/train_tweets.txt',encoding='utf-8')
pre_lines=f_test.readlines()
new_pre_lines=[]
for line in pre_lines:
    line = re.sub(" +", " ", line)
    #tokens = word_tokenize(line)
    #line = " ".join([i for i in tokens if i not in ['A','a','The','THE','the']])

    new_pre_lines.append(line)

    # array.append([id,model.predict(nl.lower().strip())[0][0][9:]])
    # print(nl.lower().strip())
    # print(model.predict((nl.lower()).strip()))
    # id+=1

lines = f.readlines()
new_lines=[]
random.shuffle(lines)
predicts = lines[:]
pos = int(0.66*len(lines))
for line2 in lines:
        line2 = re.sub(" +", " ", line2)
        arr = line2.split("\t")
        #tokens = word_tokenize(arr[1])
        #arr[1] = " ".join([i for i in tokens if i not in ['A','a','The','THE','the']])

        newText = "__label__" + arr[0] + " " + arr[1]
        new_lines.append(newText)
i=0
for i in range(15):
    f_labeled = open('new_train_tweets_all.txt', 'w', encoding='UTF-8')
    random.shuffle(new_lines)
    temLines = new_lines[:pos]
    print(pos)
    j=0
    for l in temLines:
        f_labeled.write(l + "\n")


        # if j < pos:
        # else:
        #     f_labeled2.write(newText + "\n")
        j+=1
    f_labeled.close()
    model = None
    if i%2 == 0:
        print("mod1")
        model = fasttext.train_supervised(
            input="new_train_tweets_all.txt", wordNgrams=2, ws=5, verbose=2, thread=15, t=0.0001, neg=5, minn=2,
            minCountLabel=0, minCount=1, maxn=5, lrUpdateRate=100, epoch=30, lr=2.7185326849317737, bucket=2534640, dim=107)
    elif i%2 == 1 and i<=2:
        print("mod2")
        model = fasttext.train_supervised(
            input="new_train_tweets_all.txt", wordNgrams=2, ws=5, verbose=2, thread=15, t=0.0001, neg=5, minn=3,
            minCountLabel=0, minCount=1, maxn=6, lrUpdateRate=100, epoch=30, lr=2.2389885442286235, bucket=1084286, dim=133
        )
    elif i%2 == 1:
        print("mod3")
        model = fasttext.train_supervised(
            input="new_train_tweets_all.txt", wordNgrams= 2 , ws= 5 ,verbose= 2 ,thread= 15 ,t= 0.0001 ,neg= 5 ,minn= 2 ,minCountLabel= 0 ,minCount= 1 ,maxn= 5 ,lrUpdateRate= 100 ,epoch= 40 , lr= 1.8879426931318635 , bucket= 2342751 ,dim= 251) 
    # model.save_model("model/fasttext_"+str(i)+".ftz")
    if i>5:
        pos=int(0.5*len(lines))

    def print_results(N, p, r):
        print("N\t" + str(N))
        print("P@{}\t{:.3f}".format(1, p))
        print("R@{}\t{:.3f}".format(1, r))
    # print_results(*model.test('new_train_tweets_all.txt'))
    '''
    start predict
    '''
    result=[]
    result_file = open("result/smltest_nospl_"+str(i)+".csv", "w", newline="")
    writer = csv.writer(result_file)
    writer.writerow(["Id","Predicted"])
    id = 1
    for line in new_pre_lines:
        result.append([id,model.predict(line.strip())[0][0][9:]])

    writer.writerows(result)

    result_file.close()
