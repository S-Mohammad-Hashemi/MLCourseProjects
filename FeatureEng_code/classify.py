# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 00:19:59 2014

@author: Homa Hosseinmardi
"""

from collections import defaultdict
from csv import DictReader, DictWriter

import nltk
from nltk import bigrams
from nltk import trigrams
from nltk import ngrams
import re
import numpy
import csv
from nltk.corpus import wordnet as wn
from nltk.tokenize import TreebankWordTokenizer
import nltk.classify
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn import linear_model, datasets
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
import neurolab as nl
import itertools

kTOKENIZER = TreebankWordTokenizer()

def morphy_stem(word):
    """
    Simple stemmer
    """
    stem = wn.morphy(word)
    
    if stem:
        return stem.lower()
    else:
        return word.lower()

class FeatureExtractor:
    def __init__(self):
        """
        You may want to add code here
        """
        
        None
    def preprocessing(self, text):
        #print(text)
        text = re.sub('[.!,@#$?-]', '', text)
        text = re.sub('\s(in|and|of|to|a|or|s|f|it|its|by|the|for|from|at|on)\s', ' ', text)
        text = re.sub("\s(')\s", ' ', text)
        return text
    
    def features(self, tt,T,txt):
        
        text = fe.preprocessing(txt.decode("latin1"))
        
        d1 = defaultdict(int) 
        for ii in kTOKENIZER.tokenize(text):
            #print kTOKENIZER.tokenize(text)
            d1[morphy_stem(ii)] += 1
            #print d[morphy_stem(ii)]
            
        d2 = defaultdict(int)
        fbi = list(bigrams(kTOKENIZER.tokenize(text)))
        for ii in sorted(set(fbi)):
        #for ii in (list(bigrams(kTOKENIZER.tokenize(text)))):
            d2[ii]=fbi.count(ii) 
            
        d3 = defaultdict(int)
        ftri = list(trigrams(kTOKENIZER.tokenize(text)))    
        for ii in sorted(set(ftri)):
            d3[ii]=ftri.count(ii)

        d5 = defaultdict(int)
        fn = list(ngrams(kTOKENIZER.tokenize(text),4))    
        for ii in sorted(set(fn)):
            d5[ii]=fn.count(ii)
            
        d4 = defaultdict(int)
        pos1 = nltk.pos_tag(kTOKENIZER.tokenize(text))
        for ii,jj in pos1:
            if jj!='CC' and jj!='CD' and jj!='DT' and jj!='IN' and jj!='LS' and jj!='PRP' and jj!='PRP$' and jj!='TO' and jj!='WHD' and jj!='WRB':
                d4[morphy_stem(ii)] += 1
            #print(morphy_stem(ii)[-3:])
            #d[morphy_stem(ii)[-1:]] += 1
            d4[morphy_stem(ii)[-3:]] += 1
            d4[morphy_stem(ii)[-2:]] += 1
            d4[str(len(morphy_stem(ii)))] += 1

        data = defaultdict(float)
        #del ii['Occup']
        if T==1:
            #A= ['text','id','num','cat']
            A = ['text','id','num','cat','Occup','School','Job','Achieve','Leisure','Home','Sports','TV','Music','Money','Metaph','Relig','Death','Physcal','Body','Sexual','Eating','Sleep','Groom']
            B= ['I','We','You','Negate','Assent','Optim','Insight','Discrep','Inhib','Tentat','Certain','Senses','See','Hear','Social','Comm','Othref','Friends','Family','Humans']
        else:
            A = ['id','text']
            #A = ['text','id','Occup','School','Job','Achieve','Leisure','Home','Sports','TV','Music','Money','Metaph','Relig','Death','Physcal','Body','Sexual','Eating','Sleep','Groom']
            B= ['I','We','You','Negate','Assent','Optim','Insight','Discrep','Inhib','Tentat','Certain','Senses','See','Hear','Social','Comm','Othref','Friends','Family','Humans']
            
        for kk in A:
            del tt[kk]
        #for kk in B:
         #   del tt[kk]
        data=tt
            
        d6 = dict(data.items()+d3.items()+d5.items())    
        
        return d6
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--subsample', type=float, default=1.0,
                        help='subsample this amount')
    args = parser.parse_args()
    
    # Create feature extractor (you may want to modify this)
    fe = FeatureExtractor()


    # trainOther includes all file (5183)

    C=['id','num','text','cat']
    b= numpy.array([])
    for n in C:
        first_value = []
        #print n
        trainOther1 = DictReader(open("trainOther1.csv", 'r'))
        for m in trainOther1:
            first_value = [(m[n]) for m in trainOther1]
            #print len(first_value)  
        values = []
        values.append(n)
        for ii in range(1,(len(first_value)+1)):
            values.append(first_value[ii-1])
            #o.writerow({n: values[ii-1]})
        if n == 'id':
            b = values
            #print b
        elif n == 'num' or 'cat' or 'text':
            a= numpy.array([])
            a = values
            b=numpy.c_[b,a]
    #print b
    
    #values = []
    C=['Posemo','Negemo','Occup','School','Job','Achieve','Leisure','Home','Sports','TV','Music','Money','Metaph','Relig','Death','Physcal','Body','Sexual','Eating','Sleep','Groom','Abbreviations','Emoticons','Pronoun','I','We','Self','You','Other','Negate','Assent','Affect','Optim','Anx','Anger','Sad','Cogmech','Cause','Insight','Discrep','Inhib','Tentat','Certain','Senses','See','Hear','Feel','Social','Comm','Othref','Friends','Family','Humans']
    #b= numpy.array([])
    for n in C:
        first_value = []
        #print n
        trainOther1 = DictReader(open("trainOther1.csv", 'r'))
        for m in trainOther1:
            first_value = [float(m[n]) for m in trainOther1]
         #   print len(first_value) 
        mean = []    
        mean=numpy.divide(((float(sum(first_value)))),len(first_value))
        rangee = max(first_value)-min(first_value)
        values = []
        values.append(n)
        for ii in range(1,(len(first_value)+1)):
            values.append(abs(numpy.divide((first_value[ii-1]-mean),rangee)))
            #o.writerow({n: values[ii-1]})
        #print len(values)

        a= numpy.array([])
        a = values
        #print values
        b=numpy.c_[b,a]
    #print b         
    
        
    
    # make a new file
    
    myfile = (open('train2.csv', 'wb'))
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for line in b:
        wr.writerow(line)
    myfile.close()    
    
                  



    
    # Read in training data
    #train = DictReader(open("train1.csv", 'r'))
    trainOther = DictReader(open("train2.csv", 'r'))

    # Split off dev section
    dev_train = []
    target_train = []
    dev_test = []
    target_test =[]
    full_train = []
    
    for ii in trainOther:
        
        if args.subsample < 1.0 and int(ii['id']) % 100 > 100 * args.subsample:
            continue
        
        cat=ii['cat']
        ID=ii['id']
        TXT=ii['text']
        T=1;
        feat = fe.features(ii, T,TXT)

                
        if int(ID) % 10 == 0:
            dev_test.append((feat))
            target_test.append(cat)
            
            
        else:
            dev_train.append((feat))
            target_train.append(cat)
          
        full_train.append((feat, cat))
    #print dev_test 



        
    # Train a classifier
    print("Training classifier ...")
    print len(dev_test)
    print len(dev_train)
    
    #classifier = SklearnClassifier(LinearSVC())
    #classifier.train(dev_train)
    #classifier = nltk.classify.NaiveBayesClassifier.train(dev_train)
    #classifier = nltk.classify.MaxentClassifier.train(dev_train, 'IIS', trace=3, max_iter=10)
    classifier = nl.net.newff([[-0.5, 0.5], [-0.5, 0.5]], [5, 1])
    str1 = " ".join(str(x) for x in dev_train)
    str2 = " ".join(str(x) for x in target_train)
    classifier.train(str1, str2, show=15)
    #logistic = linear_model.LogisticRegression(C=10)
    #rbm = BernoulliRBM(n_components=180, learning_rate=0.01, batch_size=10, n_iter=50, verbose=True, random_state=None)
    #clf = Pipeline(steps=[('rbm', rbm), ('clf', logistic)])
    #clf.transform(dev_train)
    #clf.fit(dev_train, target_train) 
    #Y_pred = clf.predict(dev_test)

    
   # classifier = BernoulliRBM(batch_size=10, learning_rate=0.1, n_components=2, n_iter=10,
    #   random_state=None, verbose=0)
    #print str1
    #classifier.fit(str1, str2)
    
    
    right = 0
    total = len(dev_test)
    for ii in dev_test:
        prediction = classifier.classify(ii[0])
        if prediction == ii[1]:
            right += 1
    print("Accuracy on dev: %f" % (float(right) / float(total)))
    print(nltk.classify.accuracy(classifier, dev_test))
    
    # Retrain on all data
    #classifier = nltk.classify.NaiveBayesClassifier.train(dev_train + dev_test)
    #classifier = SklearnClassifier(LinearSVC())
    #classifier.train(dev_train + dev_test)
    classifier = nltk.classify.MaxentClassifier.train(dev_train + dev_test, 'IIS', trace=3, max_iter=10)    
    # Read in test section

    o = DictWriter(open('pred.csv', 'w'), ['num', 'pred', 'cat'])
    o.writeheader()
    
    test1 = {}
    for ii in DictReader(open("testOther.csv")):
        test1[ii['num']] = ii['cat']
        #print test1
    print len(test1)
    test = {}
    for ii in DictReader(open("test1.csv")):
        T=2;
        text=ii['text']
        test[ii['num']] = classifier.classify(fe.features(ii,T,text))
        #test[ii['text']] = classifier.classify(fe.centi())
        o.writerow({'num': ii['num'], 'pred': test[ii['num']]})
    
        #o.writerow({'text': ii['txt'], 'cat': test1[ii['txt']]})
    print len(test)
        

    # Write predictions
    for ii in (test):    
        o.writerow({'num': ii, 'pred': test[ii]})
 
    print "***"
    #print test
        
       #print o.writerow({'text': ii, 'pred': test[ii]})
        


    right1 = 0
    total1 = len(test)
    for ii in test:
        
        if (test[ii]) == (test1[ii]):
            right1 += 1
            #print "x"
            #print right1
    print("Accuracy on test: %f" % (float(right1) / float(total1)))
