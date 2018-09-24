
# coding: utf-8

# In[25]:


### This is a Naive Bayes Classification Class
import nltk
from nltk import word_tokenize
import numpy as npy
import re
class MyNaiveBayes:
    def __init__(self):
    #initializing class variables
        self.classes = ["positive(+)", "negative(-)"]
        self.classNums = len(self.classes) 
        self.chanceOfOcc = {c : dict() for c in range(self.classNums) } 
        self.zvadzidza = [0 for i in range(self.classNums)]
        self.dictionaryOfWords = []
             
    
     #reads the files to be used
    def readDocs(self, document):
        retThus = []
        with open(document) as f:
            for line in f.readlines():
                this = line.split('\n')
                this = this[0].split('\t')
                revs = re.sub(r"[,/?!-()*&^%|'.,]","",this[0])
                bOw = word_tokenize(this[0].lower())
                label = int(this[1])
                retThus.append((self.mkBigram(bOw), label))
        return retThus

    def trainingDay(self, litium):
        classNumber = [0 for i in range(self.classNums)]
        lengthOf = len(litium)
        words = {c : dict() for c in range(self.classNums)}
        for book in litium:
            comment = book[0]
            col = book[-1]
            classNumber[col] += 1
            for izwi in comment:
                if izwi in words[col].keys():
                    words[col][izwi] += 1
                else:
                    words[col][izwi] = 1
                    
        for index in range(len(self.classes)):
            self.zvadzidza[index] = npy.log(classNumber[index]/lengthOf)
            self.dictionaryOfWords += list(words[index].keys())
        self.dictionaryOfWords = set(self.dictionaryOfWords)
        print (len(self.dictionaryOfWords))
        
        for index in range(len(self.classes)):
            for word in self.dictionaryOfWords:
                if word in words[index]:
                    self.chanceOfOcc[index][word] = npy.log((words[index][word]+1)/(sum(words[index].values())+len(self.dictionaryOfWords)))
                else:
                    self.chanceOfOcc[index][word] = npy.log((1)/(sum(words[index].values())+len(self.dictionaryOfWords)))

    #trains the classifier using input (type = text) referenced as books
    def trainor(self, books, test=False, split_ratio=0.30):
        litium = []
        for bk in books:
            for comment in self.readDocs(bk):
                litium.append(comment)
                
        if test:
            npy.random.shuffle(litium)
            splice = int(len(litium) * split_ratio)
            testSet = litium[:splice]
            trainSet = litium[splice:]
            self.trainingDay(trainSet)
            
            perfTest = self.testor(testSet)
            print ("You have", len(testSet)," test data")
            print ("Testing accuracy approximation: ",perfTest)
            perfTrain = self.testor(trainSet)
            print ("You have", len(trainSet)," train data")
            print ("Training accuracy approximation: ",perfTrain*100, "%")
        else:
            self.trainingDay(litium)

    
    #takes a token(s) as input and returns the numerical value of the class    
    def predictor(self, sentence):
        import operator
        tot = dict()
        for i in range(self.classNums):
            tot[i] = self.zvadzidza[i]
            for izwi in sentence:
                if izwi in self.dictionaryOfWords:
                    tot[i] = tot[i] + self.chanceOfOcc[i][izwi]
        return max(tot.items(), key=operator.itemgetter(1))[0]
    
    
    def testThisFile(self, zitaReFile):
        cols = []
        with open(zitaReFile) as f:
            for line in f.readlines():
                print(line,self.prediction(line))
                cols.append(self.prediction(line))
        
        with open('results.txt', 'w') as f:
            for cl in cols:
                print(line, self.prediction(line))
                f.write(str(cl)+"\n")    
        
                    
    #turns single words into bigrams for more efficient analysis and training           
    def mkBigram(self, words):
        data = []
        new = ['<s>'] + words + ['</s>']
        for i in range(len(new)-1):
            data.append(new[i]+'<>'+new[i+1])
        return data
    
    
    #tokenizes sentence, after which it uses the sentence to predict and assign the input to a class
    def prediction(self, text):
        sent = self.mkBigram(word_tokenize(text))
        return self.predictor(sent)
    
    def testor(self, data):
        allT = len(data)
        true = 0
        for book in data:
            reviews = book[0]
            column = book[-1]
            i = self.predictor(reviews)
            if (i == column): true += 1
        return true / allT

        
    #using already predefined labels, this function computes the accuracy of the mdel by checking the percentage of predictions that are akin to actual assignments
    def accuracy(self, output, actual):
        columnOutput = []
        actualColumns = []
        colRatio = 0
        
        with open(actual) as m:
            for line in m.readlines():
                mutupo = int(line)
                actualColumns.append(mutupo)
                
        with open(output) as m:
            for line in m.readlines():
                mutupo = int(line)
                columnOutput.append(mutupo)               


# In[26]:


classifier1 = MyNaiveBayes()
classifier1.trainor(["./Project One/sentiment_labelled_sentences/amazon_cells_labelled.txt",
                  "./Project One/sentiment_labelled_sentences/imdb_labelled.txt",
                  "./Project One/sentiment_labelled_sentences/yelp_labelled.txt"],
                 test=True,
                 split_ratio=0.30)


# In[28]:


testing12 = "this is not good at all"
classifier1.prediction(testing12)

