import sys, getopt
import numpy as np
import scipy.io.wavfile
from scikits.talkbox.features import mfcc
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from datamodelgen import createceps
import librosa
def main(argv):
	X = np.load('trainingdata.npy')
	y = np.load('trainingdatalabel.npy')
	labels = np.unique(y)
	logreg = LogisticRegression(C=1e5,verbose = 2)
	logreg.fit(X,y)
	newval = createceps(sys.argv[1])
	outp = logreg.predict(newval)
	if outp[0]==1 :
		print "Speaker is Angry"
	else :
		print "Speaker is not angry"

if __name__ == "__main__":
   main(sys.argv[1:])


