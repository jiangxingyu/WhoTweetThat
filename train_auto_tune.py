import fasttext as ft

f = open('trainDataSet',encoding='utf-8')
f_labeled = open('new_train_tweets.txt','w',encoding='UTF-8')
for line2 in f:
    arr = line2.split("\t")
    newText = "__label__" + arr[0] +" "+ arr[1]
    f_labeled.write(newText+"\n")
f_labeled.close()
f.close()

f = open('testDataSet',encoding='utf-8')
f_labeled = open('new_test_tweets.txt','w',encoding='UTF-8')
for line2 in f:
    arr = line2.split("\t")
    newText = "__label__" + arr[0] +" "+ arr[1]
    f_labeled.write(newText+"\n")
f_labeled.close()
f.close()

model = ft.train_supervised(
        input="new_train_tweets.txt", autotuneValidationFile="new_test_tweets.txt", autotuneMetric="P@1",autotuneDuration=36000
    )
model.save_model("sml_model.ftz")

print("done")