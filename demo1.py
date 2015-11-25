# Beat tracking example
from __future__ import print_function
print('reached 1')
import librosa

print('reached 1')
# 1. Get the file path to the included audio example
filename = "/Volumes/TOURO-MAC/downloads-mac/hack-the-talk-exotel-master/one.mp3" 
print('reached 1')

# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
y, sr = librosa.load(filename, duration =5.0)
print('reached 2')

# 3. Run the default beat tracker
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print('reached 3')

print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

# 4. Convert the frame indices of beat events into timestamps
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

print('Saving output to beat_times.csv')
librosa.output.times_csv('beat_times.csv', beat_times)