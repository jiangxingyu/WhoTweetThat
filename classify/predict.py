import csv
import re

import joblib

lgmodel = joblib.load("lgtest.m")
countVectorizer = joblib.load("countVectorizer.m")
filetra = open('../data/test_tweets_unlabeled.txt','r',encoding='utf-8')
allContents = filetra.readlines()
bowtraintext=[]
print(len(allContents))
for line in allContents:
    line = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''','@HTTP',line)
    # tokens = list(analyze(cc[1].strip()))
    # newtokens=[]
    # for t in tokens:
    #     nt=re.sub("\s","",t)
    #     newtokens.append(nt)
    # nline = " ".join(newtokens).strip()
    bowtraintext.append(line.strip())
features = countVectorizer.transform(bowtraintext)

f1 = open("smltest2.csv", "w", newline="")
writer = csv.writer(f1)
writer.writerow(["Id","Predicted"])
results=[]
id=1
for fe in features:
    r = lgmodel.predict(fe)
    results.append([id,r[0]])
    id+=1
writer.writerows(results)
f1.close()
