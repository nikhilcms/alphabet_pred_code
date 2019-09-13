#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:08:19 2019

@author: nikhil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re  #regex
import string
from nltk.corpus import stopwords
stopword_list = stopwords.words("english")
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,cohen_kappa_score,confusion_matrix,classification_report
bbc_data = pd.read_csv("/home/nikhil/Natural_language_processing/BBC-Dataset-News-Classification-master/dataset/dataset.csv")

doc = bbc_data.iloc[:,0]
def word_processing(doc):
    for i in range(0,len(doc)):
        doc[i] = doc[i].lower()
    for i in range(0,len(doc)):
        doc[i] = re.sub("\d+","",doc[i])
    for i in range(0,len(doc)):
        doc[i] =  doc[i].translate(string.maketrans("",""),string.punctuation)  
        doc[i] = re.sub("(<.*?>)","",doc[i])
        doc[i] = re.sub("(\\W|\\d)"," ",doc[i])
        doc[i] = doc[i].strip()
    for i in range(0,len(doc)):
        doc[i] == [ j for j in doc[i] if not j in stopword_list ]
    
    Wn = WordNetLemmatizer()
    for i in range(0,len(doc)):
        doc[i] = Wn.lemmatize(doc[i])
    ss = tuple(doc.tolist())
    return(ss)
output_process = word_processing(doc)
vect = TfidfVectorizer(stop_words = "english",min_df = 2)
X = vect.fit_transform(output_process)
Y = bbc_data.iloc[:,1]   

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

print "X_train size",X_train.shape
print "X_test",X_test.shape

model = RandomForestClassifier(n_estimators = 300,max_depth=150)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

kappa = cohen_kappa_score(y_test,y_pred)
c_mat = confusion_matrix(y_test,y_pred)
cc = accuracy_score(y_test,y_pred)
clss_report = classification_report(y_test,y_pred)

#save model to a disk
import pickle
filename1 = 'finalized_model_bbc.pkl'
pickle.dump(model, open(filename1, 'wb'))

filename2 = 'vectorizer.pkl'
pickle.dump(vect, open(filename2, 'wb'))


#load model
loaded_model = pickle.load(open("/home/nikhil/finalized_model_bbc.pkl", 'rb'))
vector_model = pickle.load(open("/home/nikhil/vectorizer.pkl", 'rb'))

test_file = open("/home/nikhil/Natural_language_processing/sport_doc.txt","r").read()
test_file = """Apple and Microsoft have announced the launch of a new iCloud app for Windows. The latest app is available for free and can be downloaded from the Microsoft Store.
It will work with all Windows 10 devices, according to a blog post on Windows Blog.The new iCloud app allows the user to access iCloud features such as iCloud Drive, Photos, Mail, Contacts, Calendar,\n 
Reminders, Safari Bookmarks and more. Now, the iCloud app has been available on Windows for quite some time" 
but it seems that the product has not been too great.n the blog post, Microsoft has said that the new iCloud app will use the same technology that is used by the company’s OneDrive’s Files On-Demand feature. 
This is quite a rare collaboration the two technology giants which are generally considered as rivals.Apple in an iCloud for Windows support page has said that the iCloud Drive on Windows 10 can initiate shared files and optimise files.
Users also pin files and documents locally. Users can also manage their iCloud storage, and upgrade it through iCloud for Windows app."""
ttest_file = pd.Series(test_file)

test_process = word_processing(ttest_file)
x = vector_model.transform(test_process)

result = loaded_model.predict(x)[0]

print "The given document is on %s related" % result


#======================Data_visualization======================#
business_data = bbc_data.iloc[1113:1824,0].tolist()
from wordcloud import WordCloud
wrdcld = WordCloud(width = 1000,height = 500,background_color="white").generate(" ".join(business_data))
plt.figure(figsize = (15,8))
plt.imshow(wrdcld)
plt.axis("off")
plt.show()
plt.savefig("sport_plot")

