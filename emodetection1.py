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
from PIL import Image
def main(argv):
	X = np.load('trainingdata2.npy')
	y = np.load('trainingdatalabel2.npy')
	labels = np.unique(y)
	logreg = LogisticRegression(C=1e5)
	logreg.fit(X,y)
	score1 = logreg.score(X,y)
	newval = createceps(sys.argv[1])
	outp = logreg.predict(newval)
	clf = svm.SVC(kernel='rbf', C = 1.0)
	clf.fit(X,y)
	score2 = clf.score(X,y)
	confidence1 = logreg.decision_function(newval)
	confidence2 = clf.decision_function(newval)
	outp2 = clf.predict(newval)
	if outp[0]==1 :
		print "Speaker is Angry"
		image = Image.open('angry.png')
		image.show()
	elif outp[0] ==2 :
		print "Speaker is scared"
		image = Image.open('scared.png')
		image.show()
	elif outp[0] ==3 :
		print "Speaker is happy"
		image = Image.open('happy.jpg')
		image.show()
	elif outp[0] ==4 :
		print "Speaker is neutral"
		image = Image.open('neutral.png')
		image.show()
	elif outp[0] ==5 :
		print "Speaker is sad"
		image = Image.open('sad.jpg')
		image.show()
		
	if outp2[0]==1 :
		print "Speaker is Angry"
		image = Image.open('angry.png')
		image.show()
	elif outp2[0] ==2 :
		print "Speaker is scared"
		image = Image.open('scared.png')
		image.show()
	elif outp2[0] ==3 :
		print "Speaker is happy"
		image = Image.open('happy.jpg')
		image.show()
	elif outp2[0] ==4 :
		print "Speaker is neutral"
		image = Image.open('neutral.png')
		image.show()
	elif outp2[0] ==5 :
		print "Speaker is sad"
		image = Image.open('sad.jpg')
		image.show()
	print "Accuracy of logistic regression %f"%score1
	print confidence1[0]
	print "Accuracy of SVM Classifier %f"%score2
	print confidence2[0]

if __name__ == "__main__":
   main(sys.argv[1:])
# def createceps(path):
# 	sample_rate, X = scipy.io.wavfile.read(path)
# 	libceps = librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=50)
# 	libceps = np.transpose(libceps)
# 	num_ceps = len(libceps)
# 	Data = []
# 	Data.append(np.mean(libceps[int(num_ceps/10):int(num_ceps*9/10)], axis =0))
# 	Data = np.array(Data)
# 	return Data


