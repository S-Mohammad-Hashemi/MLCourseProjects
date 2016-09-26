'''
Created on Mar 6, 2015

@author: Incognito
'''
import numpy as np
import cPickle, gzip
import matplotlib.pyplot as plt
import matplotlib.cm as cm
def read_data(location):
#     data = open(filename, 'r').readlines()
# 
#     y = np.array(list(int(x.split()[0]) for x in data))
#     x = np.array(list((float(x.split()[1].split(":")[1]),
#                        float(x.split()[2].split(":")[1])) for x in data))

        # Load the dataset
    f = gzip.open(location, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)

    temp_train_x, temp_train_y = train_set
    train_x=[]
    train_y=[]
    for x,y in zip(temp_train_x,temp_train_y):
        if(y==3 or y==8):
#             print y
            train_x.append(x)
            train_y.append(y)
            
#     print len(train_x),len(train_y)
    temp_test_x, temp_test_y = valid_set
    test_x=[]
    test_y=[]
    for x,y in zip(temp_test_x,temp_test_y):
        if(y==3 or y==8):
            test_x.append(x)
            test_y.append(y)
            
#     print len(test_x),len(test_y)
#     return x, y
    return train_x,train_y,test_x,test_y

# def plot_model(X, Y, clf, fignum=0):
#     # plot the line, the points, and the nearest vectors to the plane
#     plt.figure(fignum, figsize=(4, 3))
#     plt.clf()
# 
#     plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
#                 facecolors='none', zorder=10)
#     plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired)
# 
#     plt.axis('tight')
#     x_min = min(X[:, 0])
#     x_max = max(X[:, 0])
#     y_min = min(X[:, 1])
#     y_max = max(X[:, 1])
# 
#     XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
#     Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
# 
#     # Put the result into a color plot
#     Z = Z.reshape(XX.shape)
#     plt.figure(fignum, figsize=(4, 3))
#     plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
#     plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
#                 levels=[-.5, 0, .5])
# 
#     plt.xlim(x_min, x_max)
#     plt.ylim(y_min, y_max)
# 
#     plt.xticks(())
#     plt.yticks(())
# 
#     plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn import svm

    fignum = 0
    X, Y,test_x,test_y = read_data("../mnist.pkl.gz")
#     print(len(X))
#     limit_list=[5,10,15,20,50,100,200,400,800,1500,3000,6000,len(X)]
    limit_list=[1500]
    for limit in limit_list:
        index=0
        for kk, dd, gg, cc in [('linear', 0, 0, 0),]:
#                                ('poly', 1, 0, 5),
#                                ('poly', 2, 0, 5),
#                                ('poly', 3, 0, 5)]:
    #                            ('rbf', 0, 2, 0),
    #                            ('rbf', 0, 100, 0)]:
            # Fit the model
#             myC=0.08
            clf = svm.SVC(kernel=kk, degree=dd, coef0=cc, gamma=gg)
            clf.fit(X[:limit], Y[:limit])
#             print clf.support_vectors_[0] , len(clf.support_vectors_[0])
#             for i in range(10):
#                 plt.imshow(clf.support_vectors_[10*i].reshape((28, 28)), cmap = cm.Greys_r)
#                 plt.show()
#             print clf
#             limit=len(test_x)
    #         print "predictions: ", clf.predict(test_x[limit:limit+4]),test_y[limit:limit+4]
            y_predict=clf.predict(test_x)
            correct=0
#             print len(y_predict)
            for i in range(len(y_predict)):
                if y_predict[i]==test_y[i]:
                    correct+=1
            accuracy=(correct+0.0)/len(y_predict)
            accuracy=accuracy*100
            if index==0:
                print accuracy,",",
                index=1
            else:
                print accuracy,",",
                
        print ""
#         print y_predict,test_y[:limit]
#         print 
#         plot_model(X[:limit], Y[:limit], clf, fignum)
#             fignum += 1
       

    