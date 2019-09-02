# -*- coding: utf-8 -*-
import fasttext
import csv


model = fasttext.load_model("sml_model_all_v.ftz")
f = open('test_tweets_unlabeled.txt',encoding='utf-8')
stopwords = ["a","A","The","the"]

f1 = open("smltest.csv", "w", newline="")
writer = csv.writer(f1)
array = []
writer.writerow(["Id","Predicted"])
id=1
for line in f:
    words = line.split(" ")
    newWords = [each for each in words if each not in stopwords]

    nl = ' '.join(newWords)
    array.append([id,model.predict(nl.lower().strip())[0][0][9:]])
    print(nl.lower().strip())
    print(model.predict((nl.lower()).strip()))
    id+=1
    # writer.writerow([model.predict(line.strip())[0][0][9:].strip()])
writer.writerows(array)
f1.close()
f.close()
