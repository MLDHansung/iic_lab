import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import os
from os import listdir
from os.path import isfile, join
import soundfile as sf


#path='/home/iichsk/workspace/dataset/Respiratory_Sound_Database/Respiratory_Sound_Database/processed_audio_files_nopad/'
class file_Slide:

    def extract_features(self,file_name):
        max_pad_len = 100
        '''
        This function takes in the path for an audio file as a string, loads it, and returns the MFCC
        of the audio'''

        try:
            audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=2)
            #print('audio',len(audio))
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            #x = np.linspace(0, 697, mfccs.shape[1])
            #y = np.linspace(0, 40, mfccs.shape[0])
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            #mfccs = np.pad(mfccs, pad_width=((0,0)), mode='constant')
            #f = interpolate.interp2d(x, y, mfccs, kind='linear')
            #x_new = np.arange(0, 697)
            #y_new = np.arange(0, 40)
            #mfccs = f(x_new, y_new)

        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
            return None
        return mfccs

    

    # def getPureSample(self,raw_data,start,end,sr=22050):
    #     '''
    #     Takes a numpy array and spilts its using start and end args
        
    #     raw_data=numpy array of audio sample
    #     start=time
    #     end=time
    #     sr=sampling_rate
    #     mode=mono/stereo
        
    #     '''
    #     max_ind = len(raw_data) 
    #     start_ind = int(start * sr)
    #     end_ind = int(end * sr)
    #     #print('max_ind=',max_ind,'duration=',end_ind-start_ind)
        
    #     return raw_data[start_ind: end_ind]

    #os.makedirs('slide', exist_ok=True)
    def getPureSample(self,raw_data,start,end,sr=22050):
        '''
        Takes a numpy array and spilts its using start and end args
        
        raw_data=numpy array of audio sample
        start=time
        end=time
        sr=sampling_rate
        mode=mono/stereo
        
        '''
        max_ind = len(raw_data) 
        start_ind = min(int(start * sr), max_ind)
        end_ind = min(int(end * sr), max_ind)
        return raw_data[start_ind: end_ind]

    def getFilenameInfo(self, file):
        return file.split('_')

    def getSlindingfile(self, path, file_name):
        
        duration_list=[]
        feature_data1=[]
        feature_data2=[]

        audio_file_loc=path+'.wav'
        audioArr,sampleRate=librosa.load(audio_file_loc)
        duration = librosa.get_duration(y=audioArr, sr=sampleRate)
        start=0
        end=duration
        duration_list.append(end)
        c=0
        cnt = 0
        if end>=1:
            while start<=2:
                #print('start:',start,'    end:',end)
                filename= file_name + '_' + str(c) + '.wav'
                os.makedirs('/home/iichsk/workspace/respiratory_classify/slide_data/slide_slicing_data/', exist_ok=True)
                save_path='/home/iichsk/workspace/respiratory_classify/slide_data/slide_slicing_data/' + filename
                pureSample=self.getPureSample(audioArr,start,start+3,sampleRate)
                #print('pureSample',len(pureSample))

                if len(pureSample)==0:
                    #print('pureSample is empty')
                    pureSample = [0 for i in range(44100)]
                    pureSample = list(map(float,pureSample))
                
                
                #pad audio if pureSample len < max_len
                # reqLen=6*sampleRate
                # padded_data = librosa.util.pad_center(pureSample, reqLen)
                sf.write(file=save_path,data=pureSample,samplerate=sampleRate)
                if (start == 0):
                    #print('start=',start,'save_path=',save_path)
                    feature_data1 = self.extract_features(save_path)
                    #print('feature_data1',feature_data1.shape)
                    
                    # print('save mfcc image number {}...'.format(cnt))
                    # plt.figure(1)
                    # librosa.display.specshow(feature_data1, y_axis='mel',x_axis='time')
                    # #plt.colorbar(format='%+2.0f dB')
                    # plt.tight_layout()
                    # plt.savefig('{}_mfcc.png'.format(save_path))
                    # plt.close()
                elif (start　:
                    #print('start=',start,'save_path=',save_path)
                    feature_data2 = self.extract_features(save_path)
                    #print('feature_data2',feature_data2.shape)
            
                    
                    #print('save mfcc image number {}...'.format(cnt))
                    #data = feature_extraction_melspectrogram(file_name)

                    # plt.figure(1)
                    # librosa.display.specshow(feature_data2, y_axis='mel',x_axis='time')
                    # #plt.colorbar(format='%+2.0f dB')
                    # plt.tight_layout()
                    # plt.savefig('{}_mfcc.png'.format(save_path))
                    # plt.close()

                start+=1
                c+=1
        return feature_data1, feature_data2
        # print('min duration',min(duration_list))
        # print('max duration',max(duration_list))
        # sum_duration=sum(duration_list)
        # print('AVG duration',sum_duration/len(duration_list))

# class __main__():
#     file_Slide=file_Slide()
#     file_Slide.getSlindingfile(path)