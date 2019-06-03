from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

#########################################
#model = LogisticRegression(penalty= 'l2', C= 1)
#model = LinearSVC()
#model = MultinomialNB()
model =  RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)
drug= open('DrugTweets_RandomForest.txt','w')#DrugTweets_LogisticRegression,DrugTweets_SVM,DrugTweets_NaiveBayes
non_drug= open('NonDrugTweets_RandomForest.txt','w')#NonDrugTweets_LogisticRegression,NonDrugTweets_SVM,NonDrugTweets_NaiveBayes
#########################################

train_data= open('train.txt')
label_data= open('label.txt')
test_data=open('tweettext2018_tokenized.txt')

sentences=[]
y=[]
test_sentences=[]

for sentence in train_data:
	sentences.append(sentence.replace('\n',''))
for label in label_data:
	y.append(int(label.replace('\n','')))
for sentence in test_data:
	test_sentences.append(sentence.replace('\n',''))

vectorizer = CountVectorizer()
vectorizer.fit(sentences)
X_train = vectorizer.transform(sentences)
X_test = vectorizer.transform(test_sentences)

model.fit(X_train,y)
predictions = model.predict(X_test)
for i in range(len(predictions)):
	if predictions[i] == 1:
		drug.write(test_sentences[i]+'\n')
	elif predictions[i] == 0:
		non_drug.write(test_sentences[i]+'\n')

drug.close()
non_drug.close()
train_data.close()
label_data.close()
test_data.close()
