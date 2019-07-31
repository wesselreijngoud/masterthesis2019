import sys
import gensim
import argparse
import numpy as np
from random import shuffle
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

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


def levenshtein(s1, s2):
    """Levenshtein distance function adopted from: 
    https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python"""
    s1 = s1.lower()
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one
            # character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    levscore = previous_row[-1]
    return insertions, deletions, substitutions, levscore


def runLevenshtein(orig, cor):
    """A function to run actual levenshtein function and return as array"""
    levlist = []
    for i, t in zip(orig, cor):
        try:
            a, b, c, d = levenshtein(i.lower(), t)
        except TypeError:
            a, b, c, d = 0, 0, 0, 0
        levlist.append((a, b, c, d))
    levarray = np.asarray(levlist)
    return levarray


def lengthofwords(orig,cor):
    """Calculates length difference between original and correcte"""
    lengthofwords= []
    for i, t in zip(orig,cor):
        i = i.strip('|')
        t = t.strip('|')
        difflength = len(i) - len(t)
        lengthofwords.append(difflength)
    lengtharray = np.asarray(lengthofwords)
    return lengtharray

def containsApostrophe(orig, cor):
    """Checks if orig and/or cor contain apostrophes"""
    apostrophelist = []
    for i, t in zip(orig, cor):
        if "'" not in i:
            if "'" not in t:
                apostrophelist.append(0)
            elif "'" in t:
                apostrophelist.append(1)
        elif "'" in i:
            if "'" not in t:
                apostrophelist.append(2)
            elif "'" in t:
                apostrophelist.append(0)
    apostrophearray = np.asarray(apostrophelist)
    return apostrophearray


def containsSpaces(orig, cor):
    """Checks if orig and/or cor contain spaces"""
    containsSpacesList = []
    for i, t in zip(orig, cor):
        if " " not in i:
            if " " in t:
                containsSpacesList.append(1)
            else:
                containsSpacesList.append(0)
        elif " " in i:
            if " " not in t:
                containsSpacesList.append(2)
            else:
                containsSpacesList.append(0)
    containsSpacesArray = np.asarray(containsSpacesList)
    return containsSpacesArray

def wordpart(orig,cor):
    """Checks if original is part of corrected"""
    ispartofcor= []
    for i, t in zip(orig,cor):
        i = i.strip('|')
        t = t.strip('|')
        if i in t:
            ispartofcor.append(1)
        else:
            ispartofcor.append(0)
    ispartofcorarray = np.asarray(ispartofcor)
    return ispartofcorarray


def endingCheck(orig, cor):
    """Checks if endings are the same"""
    endinglist = []
    if len(orig) > 2:
        for i, t in zip(orig, cor):

            if i[-2:] != t[-2:]:
                endinglist.append(1)
            else:
                endinglist.append(0)

    endingarray = np.asarray(endinglist)
    return endingarray


def embeddingFeat(orig, cor):
    model = gensim.models.KeyedVectors.load_word2vec_format(
        'w2v.bin', unicode_errors='ignore', binary=True, limit=500000)
    simscorelist = []
    for i, t in zip(orig, cor):
        i = i.strip('|')
        t = t.strip('|')
        try:
            sim = model.similarity(i, t)
            simscorelist.append(sim)
        except KeyError:

            simscorelist.append(0)
    simscorearray = np.asarray(simscorelist)
    return simscorearray


def runAllFeatures(orig, cor):
    apostrophe = containsApostrophe(orig, cor)
    spaces = containsSpaces(orig, cor)
    levenshtein = runLevenshtein(orig, cor)
    ending = endingCheck(orig, cor)
    embeddings = embeddingFeat(orig, cor)
    partofword = wordpart(orig,cor)
    lengthwords = lengthofwords(orig,cor)
    featscomb = np.column_stack(
        (apostrophe,  spaces, ending, lengthwords, levenshtein, partofword, embeddings))
    
    return featscomb


def show_most_informative_features(vectorizer, clf, n=10):
    """function to print most informative features, obtained but edited from https://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers"""
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('trainset', metavar='trainingset',
                        type=str, help="File containing training data")
    parser.add_argument('testset', metavar='testset',
                        type=str, help="File containing test data")
    parser.add_argument(
        '-a', '--analysis',  help='Analyses wrongly classified items', action='store_true')
    parser.add_argument(
        '-r', '--results', help="Print classification report", action="store_true")
    parser.add_argument(
        '-m', '--mostinformative', help="Print most informative", action="store_true")
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

        # RUN ALL FEATURE FUNCTIONS
        trainfeats = runAllFeatures(x_train_orig, x_train_cor)
        testfeats = runAllFeatures(x_test_orig, x_test_cor)

        ######################################################################################

        # INITIALIZE THE TFIDF VECTORIZERS FOR ORIG AND COR
        vectforig = TfidfVectorizer(
            analyzer='char', ngram_range=(1, 5)).fit(x_orig)
        vectfcor = TfidfVectorizer(
            analyzer='char', ngram_range=(1, 4)).fit(x_cor)

        # TRAIN MATRIX
        trainxorig = vectforig.transform(x_train_orig).toarray()
        trainxcor = vectfcor.transform(x_train_cor).toarray()
        comb = np.column_stack((trainxorig, trainxcor, trainfeats, M))
        # TEST MATRIX
        testxorig = vectforig.transform(x_test_orig).toarray()
        testxcor = vectfcor.transform(x_test_cor).toarray()
        testcomb = np.column_stack((testxorig, testxcor, testfeats, N))

        # TRAIN ON TRAINSET
        clf.fit(comb, y_train)
        # PREDICT ON UNKNOWN SET
        y_output = clf.predict(testcomb)
        y_out_total = np.append(y_out_total, y_output)
        y_test_total = np.append(y_test_total, y_test)
        # PRINT REPORT IN DESIRED FORMAT
        if args.results:
            print('All results:')
            print(classification_report(y_test_total, y_out_total))
            print("-----------------------------")
            print("Accuracy:")
            print(accuracy_score(y_test, y_output))

            # print(confusion_matrix(y_test,y_output))

        # PRINT ANALYSIS OF WRONGLY OUTPUTTED CLASSES
        elif args.analysis:
            for i in range(len(x_test_cor)):
                if int(y_test_total[i]) != y_output[i]:
                    print(x_test_orig[i] + '\t' + x_test_cor[i] + '\t' +
                          str(int(y_test_total[i])) + '\t' + str(y_output[i]))
        elif args.mostinformative:
            print("========most informative original=========")
            show_most_informative_features(vectforig, clf)
            print("========most informative cor=========")
            show_most_informative_features(vectfcor, clf)
        else:
            parser.print_help()
