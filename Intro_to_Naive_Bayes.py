
# coding: utf-8

# # Week 8 HW by Daniel Lee
# 
# ##Goal:  Train a Naive Bayes model to classify future SMS messages as either spam or ham.
# 
# Steps:
# 
# 1.  Convert the words ham and spam to a binary indicator variable(0/1)
# 
# 2.  Convert the txt to a sparse matrix of TFIDF vectors
# 
# 3.  Fit a Naive Bayes Classifier
# 
# 4.  Measure your success using roc_auc_score
# 
# 

# In[1]:

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score


# In[2]:

df = pd.read_csv("SMSSpamCollection",sep='\t', names=['spam', 'txt'])


# In[3]:

# Initial EDA
df.head()


# In[4]:

df.describe


# In[5]:

# Convert the words ham and spam to a binary indicator variable(0/1)
# Create array of dummies
df.spam = 1 - pd.get_dummies(df.spam).apply(np.int64)


# In[6]:

# ham = 0, spam = 1
df.head()


# In[7]:

# Separate dependent variable from input feature
y = df.spam
X = df.txt


# In[8]:

#TFIDF Vectorizer: Convert the txt to a sparse matrix of TFIDF vectors
stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)
X = vectorizer.fit_transform(X)


# In[9]:

print(y.shape)
print(X.shape)


# In[10]:

# Test Train Split & Train Naive Bayes Classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
clf = naive_bayes.MultinomialNB()
clf.fit(X_train, y_train)


# In[11]:

# Model Accuracy
roc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
print ("C-stat: ", roc)

