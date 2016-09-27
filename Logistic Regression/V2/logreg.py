import random
from numpy import zeros, sign
from math import exp, log
from collections import defaultdict

import argparse
import math
import collections
import pprint

kSEED = 1701
kBIAS = "BIAS_CONSTANT"

random.seed(kSEED)


def sigmoid(score, threshold=20.0):
    """
    Prevent overflow of exp by capping activation at 20.

    :param score: A real valued number to convert into a number between 0 and 1
    """

    if abs(score) > threshold:
        score = threshold * sign(score)

    activation = exp(score)
    return activation / (1.0 + activation)


class Example:
    """
    Class to represent a logistic regression example
    """
    def __init__(self, label, words, vocab, df):
        """
        Create a new example

        :param label: The label (0 / 1) of the example
        :param words: The words in a list of "word:count" format
        :param vocab: The vocabulary to use as features (list)
        """
        self.nonzero = {}
        self.y = label
        self.x = zeros(len(vocab))
        for word, count in [x.split(":") for x in words]:
            if word in vocab:
                assert word != kBIAS, "Bias can't actually appear in document"
                self.x[vocab.index(word)] += float(count)
                self.nonzero[vocab.index(word)] = word
        self.x[0] = 1


class LogReg:
    def __init__(self, num_features, mu, step=lambda x: 0.05):
        """
        Create a logistic regression classifier

        :param num_features: The number of features (including bias)
        :param mu: Regularization parameter
        :param step: A function that takes the iteration as an argument (the default is a constant value)
        """
        
        self.beta = zeros(num_features)
        self.mu = mu
        self.step = step
        self.last_update = defaultdict(int)

        assert self.mu >= 0, "Regularization parameter must be non-negative"

    def progress(self, examples):
        """
        Given a set of examples, compute the probability and accuracy

        :param examples: The dataset to score
        :return: A tuple of (log probability, accuracy)
        """

        logprob = 0.0
        num_right = 0
        for ii in examples:
            p = sigmoid(self.beta.dot(ii.x))
            if ii.y == 1:
                logprob += log(p)
            else:
                logprob += log(1.0 - p)

            # Get accuracy
            if abs(ii.y - p) < 0.5:
                num_right += 1

        return logprob, float(num_right) / float(len(examples))

    def sg_update(self, train_example, iteration, use_tfidf=False):
        """
        Compute a stochastic gradient update to improve the log likelihood.

        :param train_example: The example to take the gradient with respect to
        :param iteration: The current iteration (an integer)
        :param use_tfidf: A boolean to switch between the raw data and the tfidf representation
        :return: Return the new value of the regression coefficients
        """
#         print("iteration is",iteration)
        
#         if iteration==0:
        if hasattr(self, "update_m") == False:
#             print("ok")
            self.update_m=[1 for i in range(len(train_example.x))]
            
#         print("update_m is:" ,self.update_m)
#         print("x in example is: ",train_example.x)
#         print("y in example is: ",train_example.y)
#         print("nonzero in example is: ",train_example.nonzero)
#         print("step is: ",self.step(0))
        sigma=0
        for t1,t2 in zip(self.beta,train_example.x):
            sigma+=t1*t2
            
        p=sigmoid(sigma, 700)
#         p=math.exp(sigma)/(1+math.exp(sigma))
        mult_landa_yp=self.step(0)*(train_example.y-p)
        reg_coeff=1-2*self.step(0)*self.mu
        for i in range(len(self.beta)):
            self.beta[i]=self.beta[i]+mult_landa_yp*train_example.x[i]
            if(train_example.x[i]==0):
                self.update_m[i]+=1
            else:
#                 print("update m is: ",self.update_m,i)
                self.beta[i]=self.beta[i]*(reg_coeff**self.update_m[i])
                self.update_m[i]=1
#                 print("update m is____: ",self.update_m,i)
#         print("self.beta is: ",self.beta)
        # TODO: Implement updates in this function

        return self.beta


def read_dataset(positive, negative, vocab, test_proportion=.1):
    """
    Reads in a text dataset with a given vocabulary

    :param positive: Positive examples
    :param negative: Negative examples
    :param vocab: A list of vocabulary words
    :param test_proprotion: How much of the data should be reserved for test
    """
    df = [float(x.split("\t")[1]) for x in open(vocab, 'r') if '\t' in x]
    vocab = [x.split("\t")[0] for x in open(vocab, 'r') if '\t' in x]
    assert vocab[0] == kBIAS, \
        "First vocab word must be bias term (was %s)" % vocab[0]

    train = []
    test = []
    for label, input in [(1, positive), (0, negative)]:
        for line in open(input):
            ex = Example(label, line.split(), vocab, df)
            if random.random() <= test_proportion:
                test.append(ex)
            else:
                train.append(ex)

    # Shuffle the data so that we don't have order effects
    random.shuffle(train)
    random.shuffle(test)

    return train, test, vocab

def step_update(iteration):
    # TODO (extra credit): Update this function to provide an
    # effective iteration dependent step size
    return 1.0

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--mu", help="Weight of L2 regression",
                           type=float, default=0.0, required=False)
    argparser.add_argument("--step", help="Initial SG step size",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--positive", help="Positive class",
                           type=str, default="../data/hockey_baseball/positive", required=False)
    argparser.add_argument("--negative", help="Negative class",
                           type=str, default="../data/hockey_baseball/negative", required=False)
    argparser.add_argument("--vocab", help="Vocabulary that can be features",
                           type=str, default="../data/hockey_baseball/vocab", required=False)
    argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=1, required=False)

    args = argparser.parse_args()
#     args.step=0.1
#     args.passes=10
#     args.mu=0.1
    train, test, vocab = read_dataset(args.positive, args.negative, args.vocab)

    print("Read in %i train and %i test" % (len(train), len(test)))

    # Initialize model
    lr = LogReg(len(vocab), args.mu, lambda x: args.step)

    # Iterations
    update_number = 0
    for pp in xrange(args.passes):
        for ii in train:
            update_number += 1
            lr.sg_update(ii, update_number)

            if update_number % 5 == 1:
                train_lp, train_acc = lr.progress(train)
                ho_lp, ho_acc = lr.progress(test)
                print("Update %i\tTP %f\tHP %f\tTA %f\tHA %f" %
                      (update_number, train_lp, ho_lp, train_acc, ho_acc))
            if update_number % 1065 == 1061:
                print "best predictors for positive class"
                for index,value in enumerate(lr.beta):
                    if(value>0.5):
                        print index,",", value,",",vocab[index]
                
                print "\nbest predictors for negative class"
                for index,value in enumerate(lr.beta):
                    if(value<-0.5):
                        print index,",", value,",",vocab[index]
                
                print "\nPoorest predictors classes"
                for index,value in enumerate(lr.beta):
                    if(value==0):
                        print index,",", value,",",vocab[index]
                print "\nPoorest predictors classes"
                for index,value in enumerate(lr.beta):
                    if(value>-0.001 and value<0):
                        print index,",", value,",",vocab[index]
                print "\nPoorest predictors for classes"
                for index,value in enumerate(lr.beta):
                    if(value<0.001 and value>0):
                        print index,",", value,",",vocab[index] 
############ this code is for finding frequency of floor(beta[i]*100) to draw histogram ################
#                 sorted_beta=sorted(lr.beta)
#                 print(sorted_beta)
#                 sorted_beta_2=[100*x for x in sorted_beta]
#                 rounded_beta=[math.floor(x) for x in sorted_beta_2]
#                 counter=collections.Counter(rounded_beta)
#                 
#                 for x in counter.items():
#                     print x[0],",",x[1]
########################################################################################################33