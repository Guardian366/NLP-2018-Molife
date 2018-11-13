
# coding: utf-8

# In[111]:


import re
import nltk
import numpy as npy
import pandas as panda
from nltk import word_tokenize
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer # term frequency-inverse document frequency (td-idf)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score #for calculating accuracy
from io import StringIO

class sentimentClass:
    
    def __init__(self, normalize=True, classifier = "logReg", split_ratio=0.3):
        #Initializes the classifier

        if classifier == "logReg":
            self.classifier = LogisticRegression(solver='lbfgs',multi_class='multinomial')
        elif classifier == "NB":
            self.classifier = naive_bayes.MultinomialNB()
            
        self.normalize = normalize
        if self.normalize:
            self.vector = TfidfVectorizer(use_idf=True)
        else:
            self.vector = TfidfVectorizer(use_idf=True, lowercase = True, stop_words = set(nltk.corpus.stopwords.words('english')), strip_accents='unicode', ngram_range=(1, 2), max_df=0.85, min_df=3, sublinear_tf=True)
            
    def curate(self, sentence):
        #creates tables of vectors which we can fit onto the data 
        return self.vector.transform(sentence.review)
    
    
    def readFile(self, files):
        #Reads all the files and creates one frame for all of them using pandas library
        info = []
        X,Y = [], []
        for x in files:
            strippedInfo = panda.read_csv(x, sep='\t', names=['review','label'])
            info.append(strippedInfo)
        info = panda.concat(info)
        self.info = info
        Y = info.label
        self.vector.fit(info.review)
        X = self.curate(info)
        
        return train_test_split(X,Y)
    
    
    def trainFunc(self, files):
        #trains the classifier using already built in libraries 
        X_train, X_test, Y_train, Y_test =  self.readFile(files)                      
        self.classifier.fit(X_train,Y_train)
        print (X_train.shape,Y_train.shape)     
        accuracy = roc_auc_score(Y_test,self.classifier.predict_proba(X_test)[:,1])
        
        #prints out the accuracy of the classification
        print ("Accuracy = ",accuracy)
        
        
    def classification(self, sentence):
        #Attempts the classification of any sentence parsed to it 
        classf = panda.read_csv(StringIO(sentence), names=['review'])
        X = self.curate(classf)
        
        #Log of probability estimates.The returned estimates for all classes are ordered by the label of classes
        Y = self.classifier.predict_proba(X)        
        return npy.argmax(Y)
    
    
    def classify(self, file):
        #classifies sentences within a file and returns a file of classifications denoted by 1 and 0
        classLabels = []
        with open(file) as f:
            for line in f.readlines():
                print(line,self.classification(line))
                labels.append(self.classification(line))
        
        with open('results.txt', 'w') as f:
            for label in classLabels:
                f.write(str(label)+"\n")
                
        print ("Results from ",file," printed to: output.txt")

        

print ("Logistic Regression Model with unnormalized sentences")
lr_u = sentimentClass(normalize=False)
lr_u.trainFunc(["./sentiment_labelled_sentences/amazon_cells_labelled.txt",
                  "./sentiment_labelled_sentences/imdb_labelled.txt",
                  "./sentiment_labelled_sentences/yelp_labelled.txt"])
print()

print ("Logistic Rregression with normalized sentences")
lr_n = sentimentClass(normalize=True)
lr_n.trainFunc(["./sentiment_labelled_sentences/amazon_cells_labelled.txt",
                  "./sentiment_labelled_sentences/imdb_labelled.txt",
                  "./sentiment_labelled_sentences/yelp_labelled.txt"])
print()

print ("Naive Bayes with unnormalized sentences")
nb_u = sentimentClass(normalize=False, classifier='NB')
nb_u.trainFunc(["./sentiment_labelled_sentences/amazon_cells_labelled.txt",
                  "./sentiment_labelled_sentences/imdb_labelled.txt",
                  "./sentiment_labelled_sentences/yelp_labelled.txt"])
print()

print ("Naive Bayes with normalized sentences")
nb_n = sentimentClass(normalize=True, classifier='NB')
nb_n.trainFunc(["./sentiment_labelled_sentences/amazon_cells_labelled.txt",
                  "./sentiment_labelled_sentences/imdb_labelled.txt",
                  "./sentiment_labelled_sentences/yelp_labelled.txt"])


# In[114]:


nb_n.classification("awefully great movie")

