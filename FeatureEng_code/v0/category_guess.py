from csv import DictReader, DictWriter

import numpy as np
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from IPython.lib.pretty import pprint
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
#             prev = " P:%s" % sentence[position - 1]
#         else:
#             prev = ""
 
#         if position < len(sentence) - 1:
#             next = "_%s" % sentence[position + 1]
# #             next=" "+word+next
#         else:
#             next = ''
 
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
#         ex += next
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
        feature_vector=feature_vector[:len(feature_vector)-1]
        yield feature_vector, y
                
        







def all_examples(limit, examples, labels, train=True):
    sent_num = 0
    for sentence,y in zip(examples,labels):
        sent_num += 1
        if train and sent_num % 5 != 0:
            if limit > 0 and sent_num > limit:
                break
            feature_vector=""
            sentence_list=sentence.split(" ")
            length=len(sentence_list)
            for jj in xrange(length):
                
                ex = example(sentence_list, jj)
                feature_vector+=ex+" "
            yield feature_vector, y
                
        
        if not train and sent_num % 5 == 0:
            if limit > 0 and sent_num > limit:
                break
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

    errors = defaultdict(int)
    for ii, ex_tuple in enumerate(examples):
        ex, correct_label = ex_tuple
        if correct_label != predictions[ii]:
            errors[(ex.split()[0], labels[predictions[ii]])] += 1

    for ww, cc in sorted(errors.items(), key=operator.itemgetter(1),
                         reverse=True)[:10]:
        print("%s\t%i" % (ww, cc))




if __name__ == "__main__":
    ####################
    ####################
    # Cast to list to keep it all in memory
    train = list(DictReader(open("train.csv", 'r')))
    test = list(DictReader(open("test.csv", 'r')))
    limit=len(train)

    feat = Featurizer()

    labels = []
    for line in train[:limit]:
        if not line['cat'] in labels:
            labels.append(line['cat'])
    texts=[x['text'] for x in train[:limit]]
     
    total_y = array(list(labels.index(x['cat']) for x in train[:limit]))
     
    list_total_train=[(ex,y) for ex,y in all_examples(limit,texts,total_y) ]
    list_total_test=[(ex,y) for ex,y in all_examples(limit,texts,total_y,train=False) ]
    list0=[ex for ex,y in list_total_train]
    print list0[0]
    list1=[ex for ex,y in list_total_test]
    x_train = feat.train_feature(list0)
    x_test = feat.test_feature(list1)
    print "her!!!!!!!!!!!!!!eeee"
#     print "here"
#     for ex, tgt in all_examples(1):
#         print(" ".join(analyzer(ex)))
#     print "end"
#### y is the index of speach tags
    list2=[y for ex, y in list_total_train]
    list3=[y for ex, y in list_total_test]
    y_train = array(list(list2))
    y_test = array(list(list3))




#     texts=[x['text'] for x in train[:limit]]    
#     texts_test =[x['text'] for x in test]
#     total_y = array(list(labels.index(x['cat']) for x in train[:limit]))
#     total_y_test=[0]*len(texts_test)
#     list_total_train=[(ex,y) for ex,y in all_examples2(texts,total_y) ]
#     list_total_test=[(ex,y) for ex,y in all_examples2(texts_test,total_y_test) ]
#     list0=[ex for ex,y in list_total_train]
#     print list0[0]
#     list1=[ex for ex,y in list_total_test]
#     x_train = feat.train_feature(list0)
#     x_test = feat.test_feature(list1)
#     print "hereeee!!!!!!!!!!!!!!"
#     print "here"
#     for ex, tgt in all_examples(1):
#         print(" ".join(analyzer(ex)))
#     print "end"
#### y is the index of speach tags
#     list2=[y for ex, y in list_total_train]
#     list3=[y for ex, y in list_total_test]
#     y_train = array(list(list2))
#     y_test = array(list(list3))



#     print y_train
    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)
 
    feat.show_top10(lr, labels)
 
#     predictions = lr.predict(x_test)
#     o = DictWriter(open("predictions.csv", 'w'), ["id", "cat"])
#     o.writeheader()
#     for ii, pp in zip([x['id'] for x in test], predictions):
#         d = {'id': ii, 'cat': labels[pp]}
#         o.writerow(d)

    print("TRAIN\n-------------------------")
    accuracy(lr, x_train, y_train,labels, list_total_train)
    print("TEST\n-------------------------")
    accuracy(lr, x_test, y_test,labels, list_total_test)







