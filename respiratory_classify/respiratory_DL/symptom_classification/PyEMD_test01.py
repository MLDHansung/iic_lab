from PyEMD import EMD
import numpy as np
import pylab as plt
from scipy.io import wavfile
import os
from os import listdir
from os.path import isfile, join
import time 
# # Define signal
# t = np.linspace(0, 1, 200)
# s = np.cos(11*2*np.pi*t*t) + 6*t*t
start = time.time()
wav = "/home/iichsk/workspace/dataset/iic_respiratory/wav_original/5_317.wav"
mypath = "/home/iichsk/workspace/dataset/iic_respiratory/wav_original/"

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))]
filepaths = [join(mypath, f) for f in filenames] # full paths of files

def EMD_filtering(s,t):
    # Execute EMD on signal
    IMF = EMD().emd(s,t)
    N = IMF.shape[0]+1
    print('IMF.shape = ',IMF.shape)
    # Plot results
    plt.figure(1)
    plt.plot(t, s, 'r')
    plt.title("iic original")
    plt.xlabel("Time [s]")
    plt.savefig('Original.png')
    plt.close()

    for n, imf in enumerate(IMF):
        plt.figure(1)
        # plt.subplot(N,1,n+2)
        plt.plot(t, imf, linewidth=0.5, color='b')
        plt.title("IMF "+ str(n+1))
        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.savefig('IFM_{}.png'.format(n))
        plt.close()


sr, x = wavfile.read(wav)

x_time = np.linspace(0, len(x)/sr, len(x))

EMD_filtering(x, x_time)
print("time :", time.time() - start)



# for file_name in filepaths:

# 	(file_dir, file_id) = os.path.split(file_name)
# 	print("dir : ", file_dir)
# 	print("name :", file_id)

	

	# fig, ax1 = plt.subplots()
	# ax1.plot(time,x,linewidth=0.5, color='b')
	# ax1.set_ylabel("Amplitude")
	# ax1.set_xlabel("Time [s]")
	# plt.title(file_id)
	# plt.axis([0,22,-35000,35000])
	# plt.grid(True)

	# plt.savefig('/home/iichsk/workspace/dataset/iic_respiratory/wavimage/'+file_id+'.png')
	# plt.colse(ax1)

	# x1 = butter_bandpass_filter(x, 120, 1800, sr, 4)


	# time2 = np.linspace(0, len(x1)/sr, len(x1))
	# fig, ax2 = plt.subplots()
	# ax2.plot(time2,x1,linewidth=0.5, color='b')
	# ax2.set_ylabel("Amplitude")
	# ax2.set_xlabel("Time [s]")
	# plt.title(file_id+' blpf')
	# plt.axis([0,22,-35000,35000])
	# plt.grid(True)
	# plt.savefig('/home/iichsk/workspace/dataset/iic_respiratory/wavimage/'+file_id+'_blpf.png')
	# #plt.colse(ax2)
	# wav_lpf = "/home/iichsk/workspace/dataset/iic_respiratory/wavfile/"+file_id+"_blpf.wav"
	# wavfile.write(wav_lpf, sr, x1.astype(np.int16))

	# (lpf_dir2, lpf_id2) = os.path.split(wav_lpf)
	# print("dir :", lpf_dir2)
	# print("name :", lpf_id2)