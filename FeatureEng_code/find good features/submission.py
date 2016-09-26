from csv import DictReader, DictWriter

import numpy as np
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
# from IPython.lib.pretty import pprint
# from sklearn.metrics.metrics import confusion_matrix
import argparse
import string
from collections import defaultdict
import operator

from numpy import array

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier #### Stochastic Gradient Classifier
from sklearn.feature_extraction.text import HashingVectorizer ### to create numerical feature vector from text
from sklearn.metrics import accuracy_score

from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from nltk.util import ngrams
import nltk
# import re
from nltk.stem.porter import PorterStemmer




class Featurizer:
    def __init__(self):
        self.vectorizer = CountVectorizer()

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        for i, category in enumerate(categories):
            top10 = np.argsort(classifier.coef_[i])[-10:]
            print("%s: %s" % (category, " ".join(feature_names[top10])))


def example(sentence, position):
#         print "@@@@@@@@@@@@@@@@@", sentence
        word = sentence[position]
        ex = word
#         tag = normalize_tags(sentence[position][1])
#         if tag in kTAGSET:
#             target = kTAGSET.index(tag)
#         else:
#             target = None

#         if position > 0:
#             prev = "%s_" % sentence[position - 1]
#             prev=prev+word+" "
#         else:
#             prev = ""
#  
        if position < len(sentence) - 1:
            next = "_%s" % sentence[position + 1]
#             next=word+next
        else:
            next = ''
 
#         all_before = " " + " ".join(["B:%s" % x
#                                      for x in sentence[:position]])
#         all_after = " " + " ".join(["A:%s" % x
#                                     for x in sentence[(position + 1):]])

#         dictionary = ["D:ADJ"] * len(wn.synsets(word, wn.ADJ)) + \
#           ["D:ADV"] * len(wn.synsets(word, wn.ADV)) + \
#           ["D:VERB"] * len(wn.synsets(word, wn.VERB)) + \
#           ["D:NOUN"] * len(wn.synsets(word, wn.NOUN))
# 
#         dictionary = " " + " ".join(dictionary)

#         char = ' '
#         padded_word = "~%s^" % sentence[position][0]
#         for ngram_length in xrange(2, 5):
#             char += ' ' + " ".join("C:%s" % "".join(cc for cc in x)
#                                    for x in ngrams(padded_word, ngram_length))
#         ex += char

#         ex += prev
        ex += next
#         ex=prev+next
#         ex += all_after
#         ex += all_before
#         ex += dictionary

        return ex





def all_examples2(examples, labels):
    sent_num = 0
    for sentence,y in zip(examples,labels):
        sent_num += 1
            
        feature_vector=""
        sentence_list=sentence.split(" ")
        length=len(sentence_list)
        for jj in xrange(length):
            
            ex = example(sentence_list, jj)
            feature_vector+=ex+" "
        yield feature_vector, y

#     good_tags=["JJ","JJR","JJS","FW","NN","NNS","NNP","NNPS","RB","RBR","RBS","VB","VBD","VBG","VBN","VBP","VBZ"]
#     tobe_list=["am","is","are","points","ten","FTP","name"]
#     sent_num = 0
#     file=open("test_features.csv",'a')
#     for sentence,y in zip(examples,labels):
#         if(sent_num % 100 == 0):
#             print sent_num
#         sent_num += 1
#         feature_vector=""
#         sentence_tokenized=nltk.word_tokenize(sentence)
#         sent_token_tag=nltk.pos_tag(sentence_tokenized)
#         good_tokens=[]
#         for word in sent_token_tag:
#             if word[1] in good_tags and word[0] not in tobe_list:
#                 good_tokens.append(word[0])
#                 file.write(word[0]+" "+word[1]+",")
#         file.write(`y`+"\n")
#         length=len(good_tokens)
#         for jj in xrange(length):
#             ex = example(good_tokens, jj)
#             feature_vector+=ex+" "
#         yield feature_vector, y
                
        


# def preprocessing(text):
#         text=text.lower()
#         #print(text)
#         text = re.sub('[.!,@#$?-]', '', text)
#         text = re.sub('\s(in|and|of|to|a|or|s|f|it|its|by|the|for|from|at|on)\s', ' ', text)
#         text = re.sub("\s(')\s", ' ', text)
#         print text
#         return text



def all_examples(examples, labels, train=True):
    sent_num = 0
    for sentence,y in zip(examples,labels):
        sent_num += 1
        if train and sent_num % 5 != 0:
            feature_vector=""
            sentence_list=sentence.split(" ")
            length=len(sentence_list)
            for jj in xrange(length):
                
                ex = example(sentence_list, jj)
                feature_vector+=ex+" "
            yield feature_vector, y
                
        
        if not train and sent_num % 5 == 0:
            feature_vector=""
            sentence_list=sentence.split(" ")
            length=len(sentence_list)
            for jj in xrange(length):
                ex = example(sentence_list, jj)
                feature_vector+=ex+" "

            yield feature_vector, y
                

def accuracy(classifier, x, y,labels, examples):
    predictions = classifier.predict(x)
    cm = confusion_matrix(y, predictions)

    print("Accuracy: %f" % accuracy_score(y, predictions))

    print("\t".join(labels))
    for ii in cm:
        print("\t".join(str(x) for x in ii))

#     errors = defaultdict(int)
#     for ii, ex_tuple in enumerate(examples):
#         ex, correct_label = ex_tuple
#         if correct_label != predictions[ii]:
#             errors[(ex.split()[0], labels[int(predictions[ii])])] += 1
#  
#     for ww, cc in sorted(errors.items(), key=operator.itemgetter(1),
#                          reverse=True)[:10]:
#         print("%s\t%i" % (ww, cc))




if __name__ == "__main__":
    # Cast to list to keep it all in memory
    total_feature_list=[]
    for i in range(10):
        name="features"+`i`+".csv"
        print name
        file=open(name,'r')
        train_partial_feature_list=[]
        for line in file:
            train_partial_feature_list.append(line)
        print len(train_partial_feature_list)
        total_feature_list+=train_partial_feature_list
    
    ########### reading test features:
    name="test_features.csv"
    print name
    file=open(name,'r')
    test_feature_list=[]
    for line in file:
        test_feature_list.append(line)
    print len(test_feature_list)
    
    train = list(DictReader(open("train.csv", 'r')))
    test = list(DictReader(open("test.csv", 'r')))

    feat = Featurizer()

    labels = []
    for line in train:
        if not line['cat'] in labels:
            labels.append(line['cat'])
    print "LENNNNNNNNNNNNNNN:",len(total_feature_list)

    texts=[x['text'] for x in train]
    test_texts=[x['text'] for x in test]
    total_y = array(list(labels.index(x['cat']) for x in train))
    test_y=[0]*len(texts) 
    list_total_train=[(ex,y) for ex,y in all_examples2(texts,total_y) ]
    list_total_test=[(ex,y) for ex,y in all_examples2(test_texts,test_y) ]
    list0_two_word=[ex for ex,y in list_total_train]
#     print list0[0]
    list1_two_word=[ex for ex,y in list_total_test]
    list0=[]
    list1=[]
    list2=[]
    list3=[]
    print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
    print len(list0_two_word),len(total_feature_list),len(list1_two_word)
#     exit(0)
    train_index=0
    test_index=0
    porter_stemmer = PorterStemmer()
#     good_tags=["JJ","JJR","JJS","FW","NN","NNS","NNP","NNPS","RB","RBR","RBS","VB","VBD","VBG","VBN","VBP","VBZ"]
    good_tags=["FW","NN","NNS","NNP","NNPS","VB","VBD","VBG","VBN","VBP","VBZ"]
    black_list=["am","is","are","points","ten","FTP","name"]
    goodfile=open("good_features.txt",'r')
    good_features=[]
    for line in goodfile:
#         label=line.split(":")[0]
#         tokens=line.split(" ")
        good_features.append(line)
        
    for index,x in enumerate(total_feature_list):
        tokens=x.split(",")
        feature_vector=""
        for i in range(len(tokens)-1):
#             if tokens[i].split(" ")[0] in good_features and labels[int(tokens[-1])]==:
            if tokens[i].split(" ")[0] in good_features[int(tokens[-1])]:
                print tokens[i].split(" ")[0]
                mult=10
            else:
                mult=1
            stem_token=porter_stemmer.stem(tokens[i].split(" ")[0])
            feature_vector+=(stem_token+" ")*mult
#         if((index+1 )% 5 != 0):
        feature_vector+=list0_two_word[train_index]
        train_index+=1
        list0.append(feature_vector)
        list2.append(tokens[-1])
#         else:
#             feature_vector+=list1_two_word[test_index]
#             test_index+=1
#             list1.append(feature_vector)
#             list3.append(tokens[-1])
    print "train index is:",train_index
    print "test index is:",test_index
#     exit(0)
    test_index=0
    for index,x in enumerate(test_feature_list):
        tokens=x.split(",")
        feature_vector=""
        for i in range(len(tokens)-1):
            if tokens[i].split(" ")[0] in good_features[int(tokens[-1])]:
                print tokens[i].split(" ")[0]
                mult=10
            else:
                mult=1
            stem_token=porter_stemmer.stem(tokens[i].split(" ")[0])
            feature_vector+=(stem_token+" ")*mult
        feature_vector+=list1_two_word[test_index] ################
        test_index+=1
        list1.append(feature_vector)
        list3.append(tokens[-1])
        
    x_train = feat.train_feature(list0)
    x_test = feat.test_feature(list1)
    print "hereeee!!!!!!!!!!!!!!"

    y_train = array(list(list2))
    y_test = array(list(list3))
 

 
    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)
  
    feat.show_top10(lr, labels)
  
    predictions = lr.predict(x_test)
    print "LLLLLLLLLLLLLLLLLL",len(predictions),len(test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "cat"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        print ii,pp
        d = {'id': ii, 'cat': labels[int(pp)]}
        o.writerow(d)
 
#     print("TRAIN\n-------------------------")
#     accuracy(lr, x_train, y_train,labels, zip(list0,list2))
#     print("TEST\n-------------------------")
#     accuracy(lr, x_test, y_test,labels, zip(list1,list3))







