import sys, getopt
import numpy as np
import scipy.io.wavfile
from scikits.talkbox.features import mfcc
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LogisticRegression
from datamodelgen2 import createceps
from sklearn import svm
import librosa
def main():
	X = np.load('trainingdata2.npy')
	y = np.load('trainingdatalabel2.npy')
	X2 = np.load('trainingdata.npy')
	y2 = np.load('trainingdatalabel.npy')
	logreg = LogisticRegression(C=1e5)
	logreg.fit(X,y)
	score1 = logreg.score(X2,y2)
	clf = svm.SVC(kernel='rbf', C = 1.0)
	clf.fit(X,y)
	score2 = clf.score(X2,y2)
	print "Accuracy of logistic regression %f"%score1
	print "Accuracy of SVM Classifier %f"%score2
if __name__ == "__main__":
   main()
