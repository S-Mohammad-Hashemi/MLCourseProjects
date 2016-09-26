import argparse
from collections import Counter, defaultdict

import random
import numpy
from numpy import median, bincount, argmax
from sklearn.neighbors import BallTree
import math
from scipy.stats.stats import itemfreq
from statsmodels.sandbox.regression.kernridgeregress_class import plt_closeall
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import collections
# from learning import sec_biggest

class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        ### 50000 ta train darim
        ### 10000 ta test darim
        ### self.train_x[0] ye vectore 784 tayist
        ### self.train_y[0] ye adadi beyne 0 ta 9 e
#         print("WWWWWWWWWWWWWWW")
#         print("train y", self.train_y[1:100])
#         print("train y", self.test_y[1:100])
#         print("test len x",len(self.test_x))
        f.close()


class Knearest:
    """
    kNN classifier
    """

    def __init__(self, x, y, k=5):
        """
        Creates a kNN instance

        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        """

        # You can modify the constructor, but you shouldn't need to
        self._kdtree = BallTree(x)
        self._y = y
        self._k = k


    def majority(self, item_indices):
        """
        Given the indices of training examples, return the majority label.  If
        there's a tie, return the median value (as implemented in numpy).

        :param item_indices: The indices of the k nearest neighbors
        """
        assert len(item_indices) == self._k, "Did not get k inputs"
        
        mylist=item_indices
        freq=collections.Counter(mylist)
#         res=bincount(mylist)
        two_most_common=freq.most_common(2);
#         biggest_index=argmax(res)
#         biggest=res[biggest_index]
        biggest_index=two_most_common[0][0]
        biggest=two_most_common[0][1]
#         print("biggest index is: ",biggest_index,biggest)
#         res[biggest_index]=0;
#         sec_biggest_index=argmax(res)
#         sec_biggest=res[sec_biggest_index]
        sec_biggest_index=two_most_common[-1][0]
        sec_biggest=two_most_common[-1][1]
#         print("sec biggest index is: ",sec_biggest_index,sec_biggest)
        res=-1
        if(biggest==sec_biggest):
#             print("go to median")
            res=numpy.median(mylist)
#             res=res-(res-math.floor(res))
        else:
#             print("without median")
            res=biggest_index
#         print("predicted label is: ",res)
        # Finish this function to return the most common y value for
        # these indices
        #
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.median.html
        #############check the most two common indexes(i,j), if i==j  pass your list to numpy.median function other wise return i
        
        
        ###################################
        ## I assume that if mylist was [1,1,2,3,3] the classify function should return 2 because of median!
        ###################################
        return res

    def classify(self, example):
        
        """
        Given an example, classify the example.

        :param example: A representation of an example in the same
        format as training data
        """
#         print("example is: ",example)
        # Finish this function to find the k closest points, query the
        ############ calculate the distance of example from all of training points. find the k values that are less than the others, create a list of k points
        ############ get the self._y[index_array[0][i]
        # majority function, and return the value.
        distance,index_array=self._kdtree.query(example,k=self._k); #### index_array[0] is the indices of k nearest neigbours.
#         print("index array is: ",index_array[0])
        mylist=[]
        for i in index_array[0]:
            mylist.append(self._y[i])
            
#         print("list is: ",mylist)
#         print("distance is:",distance)
#         print("index_array is:",index_array[0])
        
        
        return self.majority(mylist)

    def confusion_matrix(self, test_x, test_y):
        """
        Given a matrix of test examples and labels, compute the confusion
        matrixfor the current classifier.  Should return a dictionary of
        dictionaries where d[ii][jj] is the number of times an example
        with true label ii was labeled as jj.

        :param test_x: Test data representation
        :param test_y: Test data answers
        """

        # Finish this function to build a dictionary with the
        # mislabeled examples.  You'll need to call the classify
        # function for each example.

        d = defaultdict(dict)
        for i in range(0,10):
            for j in range(0,10):
                d[i][j]=0
        data_index = 0
        for xx, yy in zip(test_x, test_y):
#             print("!!!!!!!!!!!!!!!!",xx.shape) #### xx.shape = (784,)
            predicted_lable=self.classify(xx)
            ##################################################
            ### if predicted_lable was something like 4.5 I will ignore it (based on what I understand from Jordan's post on moodle)
            ##################################################
            if(predicted_lable-math.floor(predicted_lable)==0):
                d[yy][predicted_lable]+=1
            #confution_matrix[yy][classify(xx)]++ //d is confusion matrix check its value
            data_index += 1
            if data_index % 100 == 0:
                print("%i/%i for confusion matrix" % (data_index, len(test_x)))
        return d

    @staticmethod
    def acccuracy(confusion_matrix):
        """
        Given a confusion matrix, compute the accuracy of the underlying classifier.
        """

        # You do not need to modify this function

        total = 0
        correct = 0
        for ii in confusion_matrix:
            total += sum(confusion_matrix[ii].values())
            correct += confusion_matrix[ii].get(ii, 0)

        if total:
            return float(correct) / float(total)
        else:
            return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--k', type=int, default=3,
                        help="Number of nearest points to use")
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")

    # You should not have to modify any of this code
#     args.limit=100;
#     args.k=3
#     print("##############",args.limit)
    if args.limit > 0:
        print("Data limit: %i" % args.limit)
        knn = Knearest(data.train_x[:args.limit], data.train_y[:args.limit],
                       args.k)
    else:
        knn = Knearest(data.train_x, data.train_y, args.k)
    print("Done loading data")

    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    print("\t" + "\t".join(str(x) for x in xrange(10)))
    print("".join(["-"] * 90))
    for ii in xrange(10):
        print("%i:\t" % ii + "\t".join(str(confusion[ii].get(x, 0))
                                       for x in xrange(10)))
    print("Accuracy: %f" % knn.acccuracy(confusion))
