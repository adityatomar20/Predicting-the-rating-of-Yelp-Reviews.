import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve


import nltk
from nltk.corpus import stopwords

data = pd.read_csv('yelp.csv')

print(data.info())
print(data.describe())

data['text length'] = data['text'].apply(len)

g = sns.FacetGrid(data=data, col='stars')
g.map(plt.hist, 'text length', bins=50)

stars = data.groupby('stars').mean()
print(stars.corr())

data_classes = data[(data['stars']==1) | (data['stars']==3) | (data['stars']==5)]
x = data_classes['text']
y = data_classes['stars']

def text_process(text):
	nopunc = [char for char in text if char not in string.punctuation]
	nopunc = ''.join(nopunc)
	return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

vocab = CountVectorizer(analyzer=text_process).fit(x)
print(len(vocab.vocabulary_))
r0 = x[0]
print(r0)
vocab0 = vocab.transform([r0])
print(vocab0)

x = vocab.transform(x)
#Shape of the matrix:
print("Shape of the sparse matrix: ", x.shape)
#Non-zero occurences:
print("Non-Zero occurences: ",x.nnz)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=101)

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(x_train,y_train)
predmnb = mnb.predict(x_test)
print("Confusion Matrix for Multinomial Naive Bayes:")
print(confusion_matrix(y_test,predmnb))
print("Score:",round(accuracy_score(y_test,predmnb)*100,2))
print("Classification Report:",classification_report(y_test,predmnb))


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
preddt = dt.predict(x_test)
print("Confusion Matrix for Decision Tree:")
print(confusion_matrix(y_test,preddt))
print("Score:",round(accuracy_score(y_test,preddt)*100,2))
print("Classification Report:",classification_report(y_test,preddt))
