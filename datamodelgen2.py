import numpy as np
import scipy.io.wavfile
from scikits.talkbox.features import mfcc
import librosa
Data = []
Label = []
for i in range(1,127):
	sample_rate, X = scipy.io.wavfile.read("/Users/rohitgurnani/Desktop/librosademo/wav/anger/%d.wav"%i)
	libceps = librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=50)
	libceps = np.transpose(libceps)
	num_ceps = len(libceps)
	Label.append(1)
	Data.append(np.mean(libceps[int(num_ceps/10):int(num_ceps*9/10)], axis =0))
for i in range(1,69):
	sample_rate, X = scipy.io.wavfile.read("/Users/rohitgurnani/Desktop/librosademo/wav/fear/%d.wav"%i)
	libceps = librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=50)
	libceps = np.transpose(libceps)
	num_ceps = len(libceps)
	Label.append(2)
	Data.append(np.mean(libceps[int(num_ceps/10):int(num_ceps*9/10)], axis =0))
for i in range(1,71):
	sample_rate, X = scipy.io.wavfile.read("/Users/rohitgurnani/Desktop/librosademo/wav/happy/%d.wav"%i)
	libceps = librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=50)
	libceps = np.transpose(libceps)
	num_ceps = len(libceps)
	Label.append(3)
	Data.append(np.mean(libceps[int(num_ceps/10):int(num_ceps*9/10)], axis =0))
for i in range(1,79):
	sample_rate, X = scipy.io.wavfile.read("/Users/rohitgurnani/Desktop/librosademo/wav/neutral/%d.wav"%i)
	libceps = librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=50)
	libceps = np.transpose(libceps)
	num_ceps = len(libceps)
	Label.append(4)
	Data.append(np.mean(libceps[int(num_ceps/10):int(num_ceps*9/10)], axis =0))
for i in range(1,62):
	sample_rate, X = scipy.io.wavfile.read("/Users/rohitgurnani/Desktop/librosademo/wav/sadness/%d.wav"%i)
	libceps = librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=50)
	libceps = np.transpose(libceps)
	num_ceps = len(libceps)
	Label.append(5)
	Data.append(np.mean(libceps[int(num_ceps/10):int(num_ceps*9/10)], axis =0))
np.save("trainingdata2",Data)
np.save("trainingdatalabel2",Label)
def createceps(path):
	sample_rate, X = scipy.io.wavfile.read(path)
	libceps = librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=50)
	libceps = np.transpose(libceps)
	num_ceps = len(libceps)
	Data = []
	Data.append(np.mean(libceps[int(num_ceps/10):int(num_ceps*9/10)], axis =0))
	Data = np.array(Data)
	return Data