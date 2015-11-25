import numpy as np
import scipy.io.wavfile
from scikits.talkbox.features import mfcc
import librosa
Data = []
Label = []
for i in range(1,83):
	sample_rate, X = scipy.io.wavfile.read("/Volumes/TOURO-MAC/downloads-mac/hack-the-talk-exotel-master/training_dataset/angry/angrywav/ %d.wav"%i)
	libceps = librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=50)
	libceps = np.transpose(libceps)
	num_ceps = len(libceps)
	Label.append(1)
	Data.append(np.mean(libceps[int(num_ceps/10):int(num_ceps*9/10)], axis =0))
for i in range(1,22):
	sample_rate, X = scipy.io.wavfile.read("/Volumes/TOURO-MAC/downloads-mac/hack-the-talk-exotel-master/training_dataset/happy/happywav/%d.wav"%i)
	libceps = librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=50)
	libceps = np.transpose(libceps)
	num_ceps = len(libceps)
	Label.append(3)
	Data.append(np.mean(libceps[int(num_ceps/10):int(num_ceps*9/10)], axis =0))
for i in range(1,79):
	sample_rate, X = scipy.io.wavfile.read("/Volumes/TOURO-MAC/downloads-mac/hack-the-talk-exotel-master/training_dataset/unhappy/unhappywav/ %d.wav"%i)
	libceps = librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=50)
	libceps = np.transpose(libceps)
	num_ceps = len(libceps)
	Label.append(5)
	Data.append(np.mean(libceps[int(num_ceps/10):int(num_ceps*9/10)], axis =0))
np.save("trainingdata",Data)
np.save("trainingdatalabel",Label)

def createceps(path):
	sample_rate, X = scipy.io.wavfile.read(path)
	libceps = librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=50)
	libceps = np.transpose(libceps)
	num_ceps = len(libceps)
	Data = []
	Data.append(np.mean(libceps[int(num_ceps/10):int(num_ceps*9/10)], axis =0))
	Data = np.array(Data)
	return Data


