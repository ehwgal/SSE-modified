import os
from model import *
from dataloader import *
from metrics import *
from utils import *
from torch.utils.data import DataLoader

import wandb
import argparse
import yaml
import pickle
import time

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', type=str, default='/scratch/s5397774/SSE-modified/')
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--config_file',type=str, default='config.yaml')
parser.add_argument('--pretrain_clean', type=bool, default=False) # if set to True, we start training MAE from a pre-existing CAE
parser.add_argument('--pretrain_CAE', type=str, default='cae.pth')
parser.add_argument('--pretrain', type=bool, default=False)
parser.add_argument('--pretrain_MAE', type=str)
parser.add_argument('--urban_noise', type=bool, default=False)
args = parser.parse_args()

# obtain data which was unzipped to temporary local folder
tmpdir = os.environ.get('TMPDIR')
NOISE_TRAIN = "{}/urbansound16k/train/".format(tmpdir)
NOISE_TEST = "{}/urbansound16k/val/".format(tmpdir) 

# Select device
device = torch.device("cuda:0")

# Get configuration
config_file = os.path.join(args.base_dir, args.config_file)
config = get_config(config_file)
config['urban_noise'] = args.urban_noise
# Set it as a global configuration for other scripts
config["base_dir"] = args.base_dir
BASE_DIR = args.base_dir


# Setup clean, mix, test data
train_A_files, train_B_files, test_A_files, test_B_files = get_files()
train_noise_files, test_noise_files = obtain_noise_files(NOISE_TRAIN), obtain_noise_files(NOISE_TEST)

train_A_dataloader, train_B_dataloader, test_B_data = get_train_test_data(config,train_A_files,train_B_files,test_B_files,train_noise_files,test_noise_files)

# setup model
if args.pretrain:
    trainer = SSE(config).to(device)
    trainer.load_state_dict(torch.load(args.pretrain_MAE))
elif args.pretrain_clean:
    trainer = SSE(config).to(device)
    CAE = os.path.join(args.base_dir, args.pretrain_CAE)
    trainer.load_state_dict(torch.load(CAE), strict=False)
else:
    trainer = SSE(config).to(device)

stft = STFT(filter_length=config['filter_length'],hop_length=config['hop_length'],\
            win_length=config['win_length'],window=config['window']).to(device)
    
# Save model
if not os.path.exists('{}'.format(BASE_DIR)):
    os.makedirs('{}'.format(BASE_DIR))
torch.save(trainer.state_dict(), '{}/sep_trainer_init.pth'.format(BASE_DIR))
 
start_train = time.time()
wandb.init(anonymous='allow', project='audio_sep')

if not args.pretrain_clean:
    # Train the clean autoencoder
    print('Start training autoencoder A')
    for epoch in range(1,config['epochs_a']+1):
        start = time.time()
        for i, audio in enumerate(train_A_dataloader):
            loss, x_a_recon = trainer(audio.float().to(device),'a')
        #    wandb.log({'train_loss': loss})
        if epoch % 10 == 0:
            end = time.time()

            print('Epoch %d -- loss: %.3f, time: %.3f'%(epoch,torch.mean(loss.detach()).item(),end-start))

    print('Finish training autoencoder A')
    
    CAE = os.path.join(args.base_dir, args.pretrain_CAE)
    torch.save(trainer.gen_a.state_dict(), CAE)

# Train the noisy autoencoder
print('Start training autoencoder B')
for epoch in range(1,config['epochs_b']+1):
    start = time.time()
    for i, (audio_b,audio_b_clean,notnoise) in enumerate(train_B_dataloader):
        loss,mag_b,mag_b_recon,mag_ba,x_b_recon,x_ba = trainer(audio_b.float().to(device),'b',notnoise.float().to(device))
       # wandb.log({'train_loss': loss})
        if i == len(train_B_dataloader)-1 and epoch%10 == 0:
            end = time.time()

            # Log losses on the terminal
            print('Epoch %d -- loss: %.3f, time: %.3f'%(epoch, torch.mean(loss.detach()).item(),end-start))
            
            # Evaluation -- commented out as it adds to the training time
            #
            #scores_out = {'csig':[],'cbak':[],'covl':[],'pesq':[],'ssnr':[]}
            #scores_mix = {'csig':[],'cbak':[],'covl':[],'pesq':[],'ssnr':[]}
            #for i, (audio_b_test,audio_b_clean_test) in enumerate(test_B_data):
            #    print("AUDIO B TEST SHAPE: ", audio_b_test.shape)
            #    x_b_recon,mag_ba,x_ba,mag_b = trainer(audio_b_test.float().to(device),'eval')
            #    mag_ba_gt,_ = stft(audio_b_clean_test.float().to(device))
            #    add_score(eval_composite(audio_b_clean_test[0,:].float().numpy(),\
            #                          x_ba[0,:].detach().cpu().float().numpy()),scores_out)
            #    add_score(eval_composite(audio_b_clean_test[0,:].float().numpy(),\
            #                             audio_b_test[0,:].detach().cpu().float().numpy()),scores_mix)
            #    if i < 5:
            #        # Domain B original audio
            #        wandb.log({"audio_b_test-{}".format(i): [wandb.Audio(scale_audio(audio_b_test[0].cpu()).numpy(), \
            #                                         caption="input_audio_b_test-{}".format(i), sample_rate=config['sr'])]})
            #        # Domain B within domain reconstruction
            #        wandb.log({"recon_audio_b_test-{}".format(i): [wandb.Audio(scale_audio(x_b_recon[0].detach().cpu()).numpy(), \
            #                                     caption="recon_audio_b_test-{}".format(i), sample_rate=config['sr'])]})
            #        # Domain B -> domain A cross domain reconstruction
            #        wandb.log({"recon_audio_ba_test-{}".format(i): [wandb.Audio(scale_audio(x_ba[0].detach().cpu()).numpy(), \
            #                                         caption="recon_audio_ba_test-{}".format(i), sample_rate=config['sr'])]})
            #        # Domain B corresponding clean version
            #        wandb.log({"audio_ba_groundtruth_test-{}".format(i): \
            #                   [wandb.Audio(scale_audio(audio_b_clean_test[0].cpu()).numpy(),\
            #                    caption="audio_ba_groundtruth_test-{}".format(i), sample_rate=config['sr'])]})
            
            # Save model periodically
            if epoch%100 == 0:
                torch.save(trainer, '{}/sep_trainer_ep{}.pth'.format(BASE_DIR, epoch))    

            
           # avg_score_mix = avg_score(scores_mix)
           # avg_score_out = avg_score(scores_out)
           # print('Mix: ', avg_score_mix)
           # print('Out: ', avg_score_out)
                    
print('Finish training autoencoder B')
end_train = time.time()
print('Total training time: %.3f'%(end_train-start_train))

# Save model
torch.save(trainer, '{}/sep_trainer_final.pth'.format(BASE_DIR))

