from scipy.io import wavfile
from scipy import signal
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join

wav = "/home/iichsk/workspace/dataset/iic_respiratory/wav_original/5_317.wav"
mypath = "/home/iichsk/workspace/dataset/iic_respiratory/wav_original/"

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]
filepaths = [join(mypath, f) for f in filenames] # full paths of files

def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b,a,data)
	return y

for file_name in filepaths:

	(file_dir, file_id) = os.path.split(file_name)
	print("dir : ", file_dir)
	print("name :", file_id)

	sr, x = wavfile.read(file_name)

	time = np.linspace(0, len(x)/sr, len(x))

	fig, ax1 = plt.subplots()
	ax1.plot(time,x,linewidth=0.5, color='b')
	ax1.set_ylabel("Amplitude")
	ax1.set_xlabel("Time [s]")
	plt.title(file_id)
	plt.axis([0,22,-35000,35000])
	plt.grid(True)

	plt.savefig('/home/iichsk/workspace/dataset/iic_respiratory/wavimage/'+file_id+'.png')
	#plt.colse(ax1)

	'''b=signal.firwin(101, cutoff=80, fs=sr, pass_zero='lowpass')
	x1 = signal.lfilter(b, [1,0], x)
	'''

	x1 = butter_bandpass_filter(x, 120, 1800, sr, 4)


	time2 = np.linspace(0, len(x1)/sr, len(x1))
	fig, ax2 = plt.subplots()
	ax2.plot(time2,x1,linewidth=0.5, color='b')
	ax2.set_ylabel("Amplitude")
	ax2.set_xlabel("Time [s]")
	plt.title(file_id+' blpf')
	plt.axis([0,22,-35000,35000])
	plt.grid(True)
	plt.savefig('/home/iichsk/workspace/dataset/iic_respiratory/wavimage/'+file_id+'_blpf.png')
	#plt.colse(ax2)
	wav_lpf = "/home/iichsk/workspace/dataset/iic_respiratory/wavfile/"+file_id+"_blpf.wav"
	wavfile.write(wav_lpf, sr, x1.astype(np.int16))

	(lpf_dir2, lpf_id2) = os.path.split(wav_lpf)
	print("dir :", lpf_dir2)
	print("name :", lpf_id2)