import sys
import argparse
import numpy as np
from random import shuffle
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer



def loadTrainData(trainfile):
    """Loads train data file from commandline"""
    x_train_orig = []
    x_train_cor = []
    y_train = []
    trainfile = sys.argv[1]
    for line in open(trainfile):
        splitted = line.split()
        if len(splitted) < 2:
            # twid = splitted[0]
            # print(twid)
            continue
        else:
            label = int(splitted[0])
            # exclude label category 0
            if label == 0:
                continue
            # add beginning and ending character to original and normalisation
            original = '|' + splitted[1] + '|'
            norm = '|' + ' '.join(splitted[3:]) + '|'
            y_train.append(label)
            x_train_orig.append(original)
            x_train_cor.append(norm)
    # switch around labels 15 and 14
    for n, y in enumerate(y_train):
        if y == 14:
            y_train[n] = 15
        if y == 15:
            y_train[n] = 14
    return x_train_orig, x_train_cor, y_train


def loadTestData(testfile):
    """Loads test data file from command line"""
    x_test_orig = []
    x_test_cor = []
    y_test = []
    testfile = sys.argv[2]
    for line in open(testfile):
        splitted = line.split()
        if len(splitted) < 2:
            # twid = splitted[0]
            # print(twid)
            continue
        else:
            label = int(splitted[0])
            # exclude label category 0
            if label == 0:
                continue
            # add beginning and ending character to original and normalisation
            original = '|' + splitted[1] + '|'
            norm = '|' + ' '.join(splitted[3:]) + '|'
            y_test.append(label)
            x_test_orig.append(original)
            x_test_cor.append(norm)
    # switch around labels 15 and 14
    for n, y in enumerate(y_test):
        if y == 14:
            y_test[n] = 15
        if y == 15:
            y_test[n] = 14

    return x_test_orig, x_test_cor, y_test



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('trainset', metavar='trainingset',
                        type=str, help="File containing training data")
    parser.add_argument('testset', metavar='testset',
                        type=str, help="File containing test data")
    parser.add_argument(
        '-r', '--results', help="Print classification report", action="store_true")
    args = parser.parse_args()

    if args.trainset and args.testset:

        # INITIALIZE CLASSIFIER
        y_out_total = np.array([])
        y_test_total = np.array([])
        clf = LinearSVC()

        # LOAD DATASETS FROM COMMAND LINE
        x_train_orig, x_train_cor, y_train = loadTrainData(args.trainset)
        x_test_orig, x_test_cor, y_test = loadTestData(args.testset)


        # ADD ORIGS AND CORS TOGETHER FOR VECTORIZER
        x_orig = x_train_orig + x_test_orig
        x_cor = x_train_cor + x_test_cor

        ######################################################################################
        vectforig = CountVectorizer().fit(x_orig)
        vectfcor = CountVectorizer().fit(x_cor)

        # TRAIN MATRIX
        trainxorig = vectforig.transform(x_train_orig).toarray()
        trainxcor = vectfcor.transform(x_train_cor).toarray()
        # TEST MATRIX
        testxorig = vectforig.transform(x_test_orig).toarray()
        testxcor = vectfcor.transform(x_test_cor).toarray()


        # TRAIN ON TRAINSET
        clf.fit(trainxorig, y_train)
        # PREDICT ON UNKNOWN SET
        y_output = clf.predict(testxorig)
        y_out_total = np.append(y_out_total, y_output)
        y_test_total = np.append(y_test_total, y_test)
        # PRINT REPORT IN DESIRED FORMAT
        if args.results:
            print('All results:')
            print(classification_report(y_test_total, y_out_total))
            print("-----------------------------")
            print("Accuracy:")
            print(accuracy_score(y_test, y_output))

        else:
            parser.print_help()
