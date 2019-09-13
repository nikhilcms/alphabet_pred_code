#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 19:09:49 2019

@author: nikhil
"""
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from nltk.corpus import stopwords
stopwords = stopwords.words("english")
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer,LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import re
import string
lmtz = WordNetLemmatizer()
train_file = pd.read_csv("/home/nikhil/Natural_language_processing/train_E6oV3lV.csv")
test_file = pd.read_csv("/home/nikhil/Natural_language_processing/test_tweets_anuFYb8.csv")

text_data_train = train_file.tweet
text_data_test = test_file.tweet
target = train_file.label
ID = test_file.iloc[:,0]
clean = []
def text_preprocessing(text_data):
    for i in range(0,len(text_data)):
        # remove stopwords
        clean_data = ' '.join([word for word in text_data.iloc[i].split() if word not in stopwords])
        #remove other strings
	clean_data = clean_data.lower()
        reg3 = re.compile("[%s]" % re.escape('...'))
        clean_data = reg3.sub('',clean_data)
        clean_data = clean_data.decode('utf8').encode('ascii', errors='ignore')
        reg4 = re.compile("[%s]" % re.escape(string.punctuation))
        clean_data = reg4.sub('',clean_data)
        #remove digits
        clean_data = re.sub(r'\d+','',clean_data)
        clean_data = ' '.join([word for word in clean_data.split() if len(word)>2])
        clean_data = reg4.sub('',clean_data)
        clean_data = ' '.join([lmtz.lemmatize(word) for word in clean_data.split()])
        clean.append(clean_data)
    return(clean)

train_data = text_preprocessing(text_data_train)
clean = []
test_data = text_preprocessing(text_data_test)
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
#Cn = CountVectorizer()
#Tfd = TfidfTransformer(smooth_idf=False)
#txt_new = Cn.fit_transform(clean)
Tfd = TfidfVectorizer(stop_words = "english",min_df = 2,ngram_range=(1,3))
train  = Tfd.fit_transform(train_data)
test   = Tfd.transform(test_data)

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,matthews_corrcoef
from imblearn.over_sampling import RandomOverSampler,ADASYN
ros = ADASYN(random_state = 42)
train,output = ros.fit_sample(train,target)
X_train,X_test,y_train,y_test = train_test_split(train,output,test_size = 0.2)
model = RandomForestClassifier(n_estimators = 300,)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test,y_pred)
mcc = matthews_corrcoef(y_test,y_pred)
score = precision_recall_fscore_support(y_test,y_pred)

test_output = model.predict(test)
submission = pd.DataFrame(dict(label = test_output,id = ID))

final_submission = submission.to_csv("submit_twitter3.csv",encoding="utf-8",index=False)
        
plt.sa
        

