from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

train_data= open('train.txt')
label_data= open('label.txt')
fi = open('best_param1.txt','w')

sentences=[]
y=[]

for sentence in train_data:
	sentences.append(sentence.replace('\n',''))

for label in label_data:
	y.append(int(label.replace('\n','')))


def LogisticRegression_Test(TrainData,TrainLabel):
	LR = LogisticRegression(penalty= 'l2', C= 1)
	rfc_cv_score = cross_val_score(LR, TrainData, TrainLabel,verbose=False, cv = 5)
	print("=== All AUC Scores ===")
	print(rfc_cv_score)
	print('\n')
	print("=== Mean AUC Score ===")
	print("Mean AUC Score - LogisticRegression: ", rfc_cv_score.mean()) #LogisticRegression:  0.9790938606780696

def SVM_Test(TrainData,TrainLabel):
	svm = LinearSVC()
	rfc_cv_score = cross_val_score(svm, TrainData, TrainLabel,verbose=False, cv = 5)
	print("=== All AUC Scores ===")
	print(rfc_cv_score)
	print('\n')
	print("=== Mean AUC Score ===")
	print("Mean AUC Score - svm: ", rfc_cv_score.mean()) #svm:  0.9782086344710601

def RandomForest_Test(TrainData,TrainLabel):
	rfc =  RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)
	rfc_cv_score = cross_val_score(rfc, TrainData, TrainLabel,verbose=False, cv = 5)
	print("=== All AUC Scores ===")
	print(rfc_cv_score)
	print('\n')
	print("=== Mean AUC Score ===")
	print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())#Random Forest:  0.6935277181977376

def MultinomialNB_Test(TrainData,TrainLabel):
	NB = MultinomialNB()
	rfc_cv_score = cross_val_score(NB, TrainData, TrainLabel,verbose=False, cv = 5)
	print("=== All AUC Scores ===")
	print(rfc_cv_score)
	print('\n')
	print("=== Mean AUC Score ===")
	print("Mean AUC Score - MultinomialNB: ", rfc_cv_score.mean())#LogisticRegression:  0.9724904830933989

#sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.23, random_state=1000)


vectorizer = CountVectorizer()
vectorizer.fit(sentences)
X_train = vectorizer.transform(sentences)


#LogisticRegression_Test(X_train,y)
#SVM_Test(X_train,y)
RandomForest_Test(X_train,y)
#MultinomialNB_Test(X_train,y)
fi.close()




