import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils import data
from scipy.io.wavfile import read
from librosa.core import resample

import os
from pathlib import Path
import glob
import random

START = 1 # audio files start at 1st second
LEN = 2 # sample 2 sec clip
TEST_LEN = 15
EPS = 1e-8
TEST_SIZE = 50

# obtain path to clean data
tmpdir = os.environ.get('TMPDIR')
PATH_CLEAN_DIR = '{}/librispeech_selection/'.format(tmpdir)

def get_files():
    
    train_A_files = [filename for filename in glob.glob(os.path.join(PATH_CLEAN_DIR, 'train_A/*.wav'))]
    test_A_files = []

    train_B_files = [filename for filename in glob.glob(os.path.join(PATH_CLEAN_DIR, 'train_B/*.wav'))]
    test_B_files = [filename for filename in glob.glob(os.path.join(PATH_CLEAN_DIR, 'test_B/*.wav'))]

    return train_A_files, train_B_files, test_A_files, test_B_files

def obtain_noise_files(root_dir, sub_dir=None):
    """Get noise files (root_dir specifies whether they are train or test files)"""
    # get list of all wav files in main dir
    if sub_dir==None:      
        files = [filename for filename in glob.glob(os.path.join(root_dir, '**/*.wav'))]
    # get list of all wav files from specific subdir
    else:
        files = [filename for filename in glob.glob(os.path.join(root_dir, sub_dir, '*.wav'))]
    
    random.shuffle(files)
    
    return files

import soundfile as sf
# Dataset for custom noises
class DapsNoise(data.Dataset):
    def __init__(self,clean_files,noise_files,sr,clip_samples,pure_noise,snr,flag):
        self.clean_root_dir = PATH_CLEAN_DIR
        self.clean_files = clean_files
        self.noise_files = noise_files
        self.sr = sr
        self.clip_samples = clip_samples
        self.threshold = 12
        self.pure_noise = pure_noise
        self.snr_list = snr
        self.flag = flag 
        self.counter = 0
        
    def __getitem__(self,index):
        while True:
            notnoise = 1
            # Clean files
            if len(self.clean_files) != 0:
                # Randomly sample a clean file
                f = random.choice(self.clean_files)
                fs,audio = read(f)
                audio = audio.astype('float32')
                # Randomly sample a clean clip
                r = random.random()
                if r < self.pure_noise and self.flag == 'train':
                    normalized_clean = torch.zeros(LEN*self.sr).float()
                    notnoise = 0
                else:
        
                    # sometimes audio length can be too short
                    if int(len(audio)-LEN*fs) < int(START*fs): 
                        continue
                    start = random.randint(START*fs,len(audio)-LEN*fs)
                    clip = resample(audio[start:start+LEN*fs],orig_sr=fs,target_sr=self.sr)/1e5

                    if r >= self.pure_noise and np.sum(clip**2) < self.threshold and self.flag == 'train':
                        continue
                    mu, sigma = np.mean(clip), np.std(clip)
                    normalized_clean = torch.from_numpy((clip-mu)/sigma)
                
            # Noise files
            if len(self.noise_files) != 0:
                nf = random.choice(self.noise_files)
                audio_noise, fs = sf.read(nf)
                if len(audio_noise.shape) > 1:
                    audio_noise = np.mean(audio_noise,axis=1)
                audio_noise = audio_noise.astype('float32')
                # Randomly sample a clip of noise
                if len(audio_noise) < LEN*fs:
                    continue
                start = random.randint(0,len(audio_noise)-LEN*fs)
                clip_noise = resample(audio_noise[start:start+LEN*fs],orig_sr=fs,target_sr=self.sr)
                mu_noise, sigma_noise = np.mean(clip_noise), np.std(clip_noise)
                normalized_noise = torch.from_numpy((clip_noise-mu_noise)/(sigma_noise+EPS))
                
                # Mix the noise with the clean audio clip at given SNR level
                if self.counter % 3 == 0:
                    snr = self.snr_list[0]
                elif self.counter % 3 == 1:
                    snr = self.snr_list[1]
                elif self.counter % 3 == 2:
                    snr = self.snr_list[2]             

                interference = 10**(-snr/20)*normalized_noise
                # if r < self.pure_noise and self.flag == 'train':
                #     mixture = interference
                # else:
                mixture = normalized_clean + interference
                
                mu_mixture, sigma_mixture = torch.mean(mixture), torch.std(mixture)
                mixture = (mixture-mu_mixture) / sigma_mixture

                self.counter = self.counter + 1

            if len(self.noise_files) != 0:
                if self.flag == 'train':
                    return mixture, normalized_clean, notnoise 
                if self.flag == 'test':
                    return mixture, normalized_clean

            return normalized_clean

    def __len__(self):
        return 18000 # sentinel value (18000 * 2 seconds = 10 hours of training data)

class TestDapsNoise(data.Dataset): # this class was created for the test.py script
    def __init__(self,clean_files,noise_files,sr,clip_samples,pure_noise,snr,flag):
        self.clean_root_dir = PATH_CLEAN_DIR
        self.clean_files = clean_files
        self.noise_files = noise_files
        self.sr = sr
        self.clip_samples = clip_samples
        self.threshold = 12
        self.pure_noise = pure_noise
        self.snr = snr
        self.flag = flag 
        self.used_clean_files = [] # we don't want to repeat clean files for testing
        
    def __getitem__(self,index):
        while True:
            notnoise = 1
            # Clean files
            if len(self.clean_files) != 0:
                
                # select clean file that hasnt been used before
                available_files = [f for f in self.clean_files if f not in self.used_clean_files]
                # if all files are used, start over
                if len(available_files) == 0:
                    self.used_clean_files = []
                    available_files = self.clean_files
                f = random.choice(available_files)
                self.used_clean_files.append(f)

                # Load the clean audio file
                fs, audio = read(f)
                length = len(audio) / fs
                if length < 10: # audio file to be evaluated should be at least 10 seconds
                    continue
                audio = audio.astype('float32')

                # Randomly sample a clean clip
                r = random.random()
                if r < self.pure_noise and self.flag == 'train':
                    normalized_clean = torch.zeros(TEST_LEN*self.sr).float()
                    notnoise = None
                else:
                    # sometimes audio length can be too short
                    if int(len(audio)-TEST_LEN*fs) < int(START*fs): 
                        continue
                    clip = resample(audio,orig_sr=fs,target_sr=self.sr)/1e5

                    if r >= self.pure_noise and np.sum(clip**2) < self.threshold and self.flag == 'train':
                        continue
                    mu, sigma = np.mean(clip), np.std(clip)
                    normalized_clean = torch.from_numpy((clip-mu)/sigma)
                
            # Noise files
            if len(self.noise_files) != 0:
                nf = random.choice(self.noise_files)
                audio_noise, fs = sf.read(nf)
                if len(audio_noise.shape) > 1:
                    audio_noise = np.mean(audio_noise,axis=1)
                audio_noise = audio_noise.astype('float32')
                # randomly sample a clip of noise
                clip_noise = resample(audio_noise,orig_sr=fs,target_sr=self.sr)
                mu_noise, sigma_noise = np.mean(clip_noise), np.std(clip_noise)
                clip_noise = (clip_noise-mu_noise)/(sigma_noise+EPS)

                
                # if noise clip is shorter than clean clip
                if len(clip_noise) < len(normalized_clean):
                
                    # repeat the noise clip until its length matches that of the clean clip
                    num_repeats = int(np.ceil(len(normalized_clean) / len(clip_noise)))
                    clip_noise = np.tile(clip_noise, num_repeats)[:len(normalized_clean)]
                    normalized_noise = torch.from_numpy(clip_noise)
                # else noise clip NOT shorter than clean clip
                else:
                    # randomly select segment from audio
                    start_index = np.random.randint(len(clip_noise) - len(normalized_clean))
                    clip_noise = clip_noise[start_index:start_index+len(normalized_clean)]
                    normalized_noise = torch.from_numpy(clip_noise)
                
                # mix the noise with the clean audio clip at given SNR level
                interference = 10**(-self.snr/20)*normalized_noise
                
                # ensure noise in mixture is as long as clean clip
                mixture = normalized_clean + interference[:len(normalized_clean)]
                
                # normalization
                mu_mixture, sigma_mixture = torch.mean(mixture), torch.std(mixture)
                mixture = (mixture-mu_mixture) / sigma_mixture

            if len(self.noise_files) != 0: 
                if self.flag == 'test':
                    print("Created mixture!")
                    return mixture, normalized_clean

            return normalized_clean

    def __len__(self):
        return 20 # sentinel value

# Get the dataloader for clean, mix, and test
def get_train_test_data(config,train_A_files,train_B_files,test_B_files,train_noise_files=None,test_noise_files=None):
    # Clean
    train_A_data = DapsNoise(train_A_files,[],config['sr'],config['clip_size'],config['pure_noise_a'],config['snr'],'train')
    # Noisy train
    train_B_data = DapsNoise(train_B_files,train_noise_files,config['sr'],config['clip_size'],\
                                                    config['pure_noise_b'],config['snr'],'train')
    # Noisy test
    test_B_data = DapsNoise(test_B_files,test_noise_files,config['sr'],config['clip_size'],\
                                                    config['pure_noise_b'],config['snr'],'test')

    train_A_dataloader = DataLoader(train_A_data, batch_size=config['b_size'], shuffle=True, \
                                    num_workers=config['num_workers'], drop_last=True, pin_memory=True)
    train_B_dataloader = DataLoader(train_B_data, batch_size=config['b_size'], shuffle=True, \
                                    num_workers=config['num_workers'], drop_last=True, pin_memory=True)
    test_B_dataloader = DataLoader(test_B_data, batch_size=1, shuffle=True, pin_memory=True)
    
    test_B_data = []
    for i, audio_pair in enumerate(test_B_dataloader):
        test_B_data.append(audio_pair)
    return train_A_dataloader, train_B_dataloader, test_B_data
