# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 00:29:41 2021

@author: 1013449
"""


import csv
import string
import re
import scipy.sparse
import numpy as np

# List of  document class
l_cl = []

# List of title + abstract
l_tabs = []


###################### Read the file #####################################

#Save document class and title + abstract
with open('abstractdata5.csv', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='#')
    line_count = 0
    for row in csv_reader:
        if line_count == 43:
            a = 1
        l_cl.append(int(row[1]))
        str_tabs = row[2]+" "+row[3]
        l_tabs.append(str_tabs)
        line_count = line_count+1
        

########### PREPROCESSING ####################



### STOP WORDS USING SKLEARN LIBRARY
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer



########### STEMMING/LEMMATIZING FUNCTION ########################
def stemming(tokens):
    #Get words   
    stemmer = SnowballStemmer(language='english')
    for i in range(0, len(tokens)):
        tokens[i] = stemmer.stem(tokens[i])
    return " ".join(tokens)  
    
def lemmatizing(tokens):
    #Lemmatizer 
    lemmatizer = WordNetLemmatizer()
    for i in range(0, len(tokens)):
        tokens[i] = lemmatizer.lemmatize(tokens[i])
    return " ".join(tokens)  
    
  


 
for i in range(0,len(l_tabs)):
    
    #################################### SIMPLE PREPROCESSING #####################
    #Lower case
    l_tabs[i] = l_tabs[i].lower()
    #Replace character - with space
    l_tabs[i]=l_tabs[i].replace('-',' ')
    #Remove punctuation
    l_tabs[i] = l_tabs[i].translate(str.maketrans(' ', ' ', string.punctuation))
    #Remove chinese characters 
    RE = re.compile(u'[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]', re.UNICODE)
    l_tabs[i] = RE.sub('', l_tabs[i])
    #Remove links, words beginning with https, www, url
    l_tabs[i] = re.sub('(?:\s)https[^, ]*', '', l_tabs[i])
    l_tabs[i] = re.sub('(?:\s)http[^, ]*', '', l_tabs[i])
    l_tabs[i] = re.sub('(?:\s)www[^, ]*', '', l_tabs[i])
    l_tabs[i] = re.sub('(?:\s)url[^, ]*', '', l_tabs[i])
    
    #Remove numbers
    l_tabs[i] = re.sub(r'\b[0-9]+\b\s*', '', l_tabs[i])
    
    #Remove words starting with a number
    #l_tabs[i] = re.sub('\\b\\d[^,]*(?:,|$)', '', l_tabs[i])
    
    #Remove double space
    l_tabs[i] = re.sub(' +', ' ',l_tabs[i])
    
    #################################### REMOVING STOPWORDS #####################
    words = [word for word in l_tabs[i].split() if word.lower() not in ENGLISH_STOP_WORDS]
   
    
    #################################### STEMMING  #####################
    l_tabs[i] = stemming(words)
     
     
    
    
####################### TFID transformation #################################
from sklearn.feature_extraction.text import CountVectorizer

########### USING SKLEARN 
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer2 = TfidfVectorizer()
m_TFIDSKLEARN_sparse = vectorizer2.fit_transform(l_tabs)

#To array
m_TFIDSKLEARN = m_TFIDSKLEARN_sparse.toarray()

############ BASIC TFID
vectorizer = CountVectorizer()
m_CVsparse = vectorizer.fit_transform(l_tabs)
#print(vectorizer.get_feature_names()) #NAMES
m_CV = m_CVsparse.toarray()  #To array

## Matrix to obtain tfid
m_TFID = np.zeros(m_CV.shape)

n = m_CV.shape[0]  #Number of documents

for i in range(0, m_CV.shape[0]):
    for j in range(0, m_CV.shape[1]):
        if(m_CV[i,j] != 0):
            tf = m_CV[i,j]/sum(m_CV[i])
            nj = np.count_nonzero(m_CV[:,j])
            m_TFID[i,j] = tf*np.log(n/nj)
        
########################### K- MEANS ################################# 
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5).fit(m_TFID)
kmeans_Sklearn = KMeans(n_clusters=5).fit(m_TFIDSKLEARN)

########################### NMI ################################# 
from sklearn.metrics.cluster import normalized_mutual_info_score

NMI = normalized_mutual_info_score(l_cl, kmeans.labels_, average_method = 'geometric')
NMI_Sklearn = normalized_mutual_info_score(l_cl, kmeans_Sklearn.labels_, average_method = 'geometric')
print('The NMI is ', NMI)
print('The NMI using TFID from SKLearn is ', NMI_Sklearn)

######################## KEYWORDS #####################################
for i in range(0,5):
    words_labels = m_CV[kmeans_Sklearn.labels_==i, :]
    sumWords = np.sum(words_labels,axis=0)
    n=10 ###Number of words
    temp = np.argpartition(-sumWords, n )
    argmax_words = temp[:n]
   
    for j in range(0, len(argmax_words)):
        print('Label ', i, 'word: ', vectorizer.get_feature_names()[argmax_words[j]])

