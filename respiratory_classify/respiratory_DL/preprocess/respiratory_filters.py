from scipy.io import wavfile
from scipy import signal
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import pywt
import padasip as pa 
from math import log10, sqrt

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

def cheby1_filter(data, lowcut, highcut, fs, order=5, riple=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low == 0 and high == 1:
        return data
    elif low == 0 and high != 1:
        b, a = signal.cheby1(order, riple, highcut / nyq, btype='low')
    elif low != 0 and high == 1:
        b, a = signal.cheby1(order, riple, lowcut / nyq, btype='high')
    elif low != 0 and high != 1:
        b, a = signal.cheby1(order, riple, [low, high], btype='band')
    output = signal.filtfilt(b, a, data)
    return output

def wavelet_dwt(data, thresh = 0.63, wavelet="db4"):
    thresh = thresh*np.nanmax(data)
    coeff = pywt.wavedec(data, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal

# def lms(x, d, N = 4, mu = 0.1):
#     nIters = min(len(x),len(d)) - N
#     u = np.zeros(N)
#     w = np.zeros(N)
#     e = np.zeros(nIters)
#     for n in range(nIters):
#         u[1:] = u[:-1]
#         u[0] = x[n]
#         e_n = d[n] - np.dot(u, w)
#         w = w + mu * e_n * u
#         e[n] = e_n
#     return e

# def adaptive_LMS(d,x2_signal):
  
#     #LMS Filter:
#     f_lms = pa.filters.FilterLMS(n=1, mu=0.1, w="random")
#     y, e, w = f_lms.run(d, x2_signal)
    
#     mse_LMS = np.mean(e**2)
#     print('MSE LMS = ', mse_LMS)
#     psnr_LMS = 20*log10(2/sqrt(mse_LMS))
#     print('PSNR for LMS = ',psnr_LMS)

#     return y
def lms(x, d, N = 4, mu = 0.1):
    nIters = min(len(x),len(d)) - N
    u = np.zeros(N)
    w = np.zeros(N)
    e = np.zeros(nIters)
    for n in range(nIters):
        u[1:] = u[:-1]
        u[0] = x[n]
        e_n = d[n] - np.dot(u, w)
        w = w + mu * e_n * u
        e[n] = e_n
    return e

for file_name in filepaths:
    #### original data
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

    #### chevyshev type 1 filter
    x1 = cheby1_filter(x, 15, 1700, sr, 4)
    time2 = np.linspace(0, len(x1)/sr, len(x1))
    fig, ax2 = plt.subplots()
    ax2.plot(time2,x1,linewidth=0.5, color='b')
    ax2.set_ylabel("Amplitude")
    ax2.set_xlabel("Time [s]")
    plt.title(file_id+' chev')
    plt.axis([0,22,-35000,35000])
    plt.grid(True)
    plt.savefig('/home/iichsk/workspace/dataset/iic_respiratory/wavimage/'+file_id+'_chev.png')
    wav_lpf = "/home/iichsk/workspace/dataset/iic_respiratory/wavfile/"+file_id+"_chev.wav"
    wavfile.write(wav_lpf, sr, x1.astype(np.int16))
    (lpf_dir2, lpf_id2) = os.path.split(wav_lpf)
    print("dir :", lpf_dir2)
    print("name :", lpf_id2)

    #### wavelet dwt filter
    x2 = wavelet_dwt(x1, 0.4)
    time3 = np.linspace(0, len(x2)/sr, len(x2))
    fig, ax3 = plt.subplots()
    ax3.plot(time3,x2,linewidth=0.5, color='b')
    ax3.set_ylabel("Amplitude")
    ax3.set_xlabel("Time [s]")
    plt.title(file_id+' chev + dwt')
    plt.axis([0,22,-35000,35000])
    plt.grid(True)
    plt.savefig('/home/iichsk/workspace/dataset/iic_respiratory/wavimage/'+file_id+'_dwt.png')
    wav_lpf = "/home/iichsk/workspace/dataset/iic_respiratory/wavfile/"+file_id+"_dwt.wav"
    wavfile.write(wav_lpf, sr, x2.astype(np.int16))
    (lpf_dir3, lpf_id3) = os.path.split(wav_lpf)
    print("dir :", lpf_dir3)
    print("name :", lpf_id3)
    plt.close()
    #### adaptive filter
    noise = 1  # * swing
    N = len(x2)
    dim = 1
    x2_signal = np.reshape(x2, (N//dim, dim)) # input matrix
    v = np.random.normal(scale=noise, size=N)# define noise matrix
    d = x2_signal[:,0] + v # add noise to target matrix
    f_lms = pa.filters.FilterNLMS(n=dim, mu=0.1, w="random")
    x3, e, w = f_lms.run(d, x2_signal)
    print(x2-x3)
    x4 = x2-x3
    time4 = np.linspace(0, len(x3)/sr, len(x3))
    fig, ax4 = plt.subplots()
    ax4.plot(time4, x3,linewidth=0.5, color='b')
    ax4.set_ylabel("Amplitude")
    ax4.set_xlabel("Time [s]")
    plt.title(file_id+'dwt data - lms data')
    plt.axis([0,22,-35000,35000])
    plt.grid(True)
    plt.savefig('/home/iichsk/workspace/dataset/iic_respiratory/wavimage/'+file_id+'_lms.png')
    wav_lpf = "/home/iichsk/workspace/dataset/iic_respiratory/wavfile/"+file_id+"_lms.wav"
    wavfile.write(wav_lpf, sr, x3.astype(np.int16))
    (lpf_dir4, lpf_id4) = os.path.split(wav_lpf)
    print("dir :", lpf_dir4)
    print("name :", lpf_id4)




