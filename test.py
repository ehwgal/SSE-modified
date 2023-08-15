
# Test script written by Ellemijn Galjaard, largely based on train.py in the original SSE project (credit to Wang et al.)

from model import *
from dataloader import *
from metrics import *
from utils import *
from torch.utils.data import DataLoader

import wandb
import io
import os
import math
import argparse
import yaml
import pickle
import time
import soundfile as sf
import torch
import random
import numpy as np

# make sure test results are the same every time
# by setting a random seed etc.
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, default='/scratch/s5397774/SSE-modified/')
parser.add_argument('--config_file',type=str, default='config.yaml')
parser.add_argument('--trained_model', type=str, default='BASELINE.pth')
parser.add_argument('--test_snr', type=int, default=None) # choose the SNR to test for: -5, 0, or 5 if you are creating your own test set
parser.add_argument('--subdir', type=str, default=None) # choose the data class to test for in your set
parser.add_argument('--padding', type=bool, default=True) # decide whether to add padding to audio
parser.add_argument('--data_available', type=bool, default=True) # turn to False if you want to create your own test set
parser.add_argument('--chosen_file', type=str, default='climbing_children_snr_minus5.pkl') # path to created test set
args = parser.parse_args()

tmpdir = os.environ.get('TMPDIR')
# Select device
device = torch.device("cpu")

# test set to select noise from
TEST_SET = "{}/gymnoise_testing/".format(tmpdir)

# Get configuration
config_file = os.path.join(args.base_dir, args.config_file)
config = get_config(config_file)
config['urban_noise'] = args.urban_noise
BASE_DIR = args.base_dir

trainer = torch.load(args.trained_model, map_location=device)

if args.data_available == False:
    # get clean speech files and noise files
    train_A_files, train_B_files, test_A_files, test_B_files = get_files()
    test_noise_files = obtain_noise_files(TEST_SET, sub_dir=args.subdir)
    print("CHECK:\n", test_noise_files[0], "\n", test_noise_files[1])

    # load test data and create mixtures
    test_dapsnoise = TestDapsNoise(test_B_files, test_noise_files, config['sr'],config['clip_size'],config['pure_noise_b'],args.test_snr,'test')
    test_dataloader = DataLoader(test_dapsnoise, batch_size=1, shuffle=False, pin_memory=True)

    test_data = []
    for i, audio_pair in enumerate(test_dataloader):
        test_data.append(audio_pair)

    with open('/scratch/s5397774/SSE-modified/TRYOUT_MUSIC_5.pkl', 'wb') as file:
        pickle.dump(test_data, file)
else:
    with open('{}/{}'.format(BASE_DIR, args.chosen_file), 'rb') as file:
        print("TESTING ON: \t", args.chosen_file)
        test_data = pickle.load(file)

print("Obtained test data!")
#print("Current test snr: ", int(args.test_snr))


# Evaluation

def process_full_audio(trainer, mixture_test, window_size=2, stride=2):
    
    audio_length = len(mixture_test.reshape(-1)) 
    output_audio = []

    # calculate the number of windows needed to cover the entire audio
    num_windows = math.floor(audio_length / (2*config['sr']))
    padding_amount = (2*config['sr']) - (audio_length % (2*config['sr']))

    if padding_amount != 0:
        num_windows = num_windows + 1

    padding = torch.zeros(padding_amount) # pad the tensor for any remaining audio 
    padding = padding.unsqueeze(0)

    total_test = torch.cat([mixture_test, padding], dim=1)
        
    # predict enhanced version of chunks in loop
    for i in range(num_windows):
        start_sample = i * stride * config['sr']
        end_sample = min(start_sample + window_size * config['sr'], len(total_test.reshape(-1)))
        audio_chunk = total_test[:, start_sample:end_sample]
             
        _, _, pred_enhanced, _ = trainer(audio_chunk.float().to(device), 'eval')
        
        output_audio.append(pred_enhanced)    

    # concatenate all tensors together
    enhanced_audio = torch.cat(output_audio, dim=1)
    
    # normalization
    #mu_audio, sigma_audio = torch.mean(x_ba_full), torch.std(x_ba_full)
    #enhanced_audio = (enhanced_audio - mu_audio) / sigma_audio


    return enhanced_audio, padding

scores_out = {'csig':[],'cbak':[],'covl':[],'pesq':[],'ssnr':[], 'nisqa':[], 'stoi':[]} # add stoinet measure
scores_mix = {'csig':[],'cbak':[],'covl':[],'pesq':[],'ssnr':[], 'nisqa':[], 'stoi':[]}
# cbak csig covl pesq stoi and ssnr are intrusive measures
# nisqa and stoinet are non-intrusive measures and model-based

wandb.init()

for i, (mixture, clean) in enumerate(test_data[0]):
   
    # obtain model prediction (enhanced audio)
    enhanced, zero_padding = process_full_audio(trainer, mixture)  
    print("ENHANCED DIMENSIONS: ", enhanced.shape)
    print("MIXTURE DIMENSIONS: ", mixture.shape)

    if args.padding:
        # make clean and mixture same size as enhanced by adding padding
        # (csig and covl will not be calculated due to zeroes)
        clean = torch.cat([clean, zero_padding], dim=1)
        mixture = torch.cat([mixture, zero_padding], dim=1)

    else:
        # else, make enhanced same size as clean/mixture by cropping
        enhanced = enhanced[:, :len(mixture.reshape(-1))]

    #print("length mixture: ", len(mixture.reshape(-1)) / 16000, "s")
    #print("length clean: ", len(clean.reshape(-1)) / 16000, "s")
    #print("length enhanced: ", len(enhanced.reshape(-1)) / 16000, "s")

    add_score(eval_composite(clean[0,:].float().numpy(),\
                                    enhanced[0,:].detach().cpu().float().numpy(), mixture=False),scores_out) # score added for enhanced
    add_score(eval_composite(clean[0,:].float().numpy(),\
                                   mixture[0,:].detach().cpu().float().numpy(), mixture=True),scores_mix) # scor
avg_score_mix = avg_score(scores_mix)
avg_score_out = avg_score(scores_out)
print('Score mixture: ', avg_score_mix)
print('Score enhanced: ', avg_score_out)
