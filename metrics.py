''' 
Reference:
https://github.com/santi-pdp/segan_pytorch/blob/master/segan/utils.py
https://github.com/svj1991/Adaptive_front_ends/blob/master/sepcosts.py
'''

import subprocess
from subprocess import run, PIPE
from scipy.linalg import toeplitz
from scipy.io import wavfile
# from numba import jit, int32, float32
import soundfile as sf
from scipy.signal import lfilter
from scipy.interpolate import interp1d
import glob
import numpy as np
import tempfile
import os
import random
import re
import string

import soundfile as sf
from pystoi import stoi
import torch
import torch.nn as nn
import torch.nn.functional as F
from stft import *
from pesq import pesq, PesqError

EPS = 1e-8

def eval_composite(clean_utt, Genh_utt, mixture=True, base_dir=None, noisy_utt=None):
    clean_utt = clean_utt.reshape(-1)
    Genh_utt = Genh_utt.reshape(-1)
    csig, cbak, covl, pesq, ssnr, nisqa, stoi = CompositeEval(clean_utt,
                                                 Genh_utt, mixture, base_dir, True)
    evals = {'csig':csig, 'cbak':cbak, 'covl':covl,
            'pesq':pesq, 'ssnr':ssnr, 'nisqa':nisqa, 'stoi':stoi}
    if noisy_utt is not None:
        noisy_utt = noisy_utt.reshape(-1)
        csig, cbak, covl, \
        pesq, ssnr, nisqa, stoi = CompositeEval(clean_utt,
                                   noisy_utt,
                                   True, base_dir)
        return evals, {'csig':csig, 'cbak':cbak, 'covl':covl,
                'pesq':pesq, 'ssnr':ssnr, 'nisqa':nisqa, 'stoi':stoi}
    else:
        return evals

def STOI_calculate(ref_wav, deg_wav):
    # calculate short-time objective intelligibility score

    tfl = tempfile.NamedTemporaryFile()
    ref_tfl = tfl.name + '_ref.wav'
    deg_tfl = tfl.name + '_deg.wav'

    sf.write(ref_tfl, ref_wav, 16000, subtype='PCM_16')
    sf.write(deg_tfl, deg_wav, 16000, subtype='PCM_16')

    curr_dir = os.getcwd()
    rate, ref = wavfile.read(ref_tfl)
    rate, deg = wavfile.read(deg_tfl)

    # ref and deg should have the same length, and be 1D
    stoi_score = stoi(ref, deg, 16000)
   

    return stoi_score

def NISQA(deg_wav, base_dir):
    # NISQA does not require a reference

    tfl = tempfile.NamedTemporaryFile()
    deg_tfl = tfl.name + '_deg.wav'

    sf.write(deg_tfl, deg_wav, 16000, subtype='PCM_16')

    curr_dir = os.getcwd()

    os.chdir('{}NISQA'.format(base_dir))

    # get NISQA score
    output = subprocess.check_output([f"python run_predict.py --mode predict_file --pretrained_model weights/nisqa_tts.tar --deg {deg_tfl}"], shell=True)

    # format score
    output_string = output.decode('utf-8')
    nisqa_score = float(output_string.split()[-1])
    
    os.chdir(curr_dir)

    return nisqa_score

def PESQ(ref_wav, deg_wav, mixture):
    # reference wav is the clean file, while the degraded wav is either the mixture or the enhanced prediction

    tfl = tempfile.NamedTemporaryFile()
    ref_tfl = tfl.name + '_ref.wav'
    deg_tfl = tfl.name + '_deg.wav'

    sf.write(ref_tfl, ref_wav, 16000, subtype='PCM_16')
    sf.write(deg_tfl, deg_wav, 16000, subtype='PCM_16')

    curr_dir = os.getcwd()
    rate, ref = wavfile.read(ref_tfl)
    rate, deg = wavfile.read(deg_tfl)
    # Write both to tmp files and then eval with pesqmain
    pesq_score = pesq(rate, ref, deg, mode='wb', on_error=PesqError.RETURN_VALUES)
    return pesq_score



def SSNR(ref_wav, deg_wav, srate=16000, eps=1e-10):
    """ Segmental Signal-to-Noise Ratio Objective Speech Quality Measure
        This function implements the segmental signal-to-noise ratio
        as defined in [1, p. 45] (see Equation 2.12).
    """
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]
   
    # scale both to have same dynamic range. Remove DC too.
    dif = ref_wav - deg_wav
    overall_snr = 10 * np.log10(np.sum(ref_wav ** 2) / (np.sum(dif ** 2) +
                                                        10e-20))
    # global variables
    winlength = int(np.round(30 * srate / 1000)) # 30 msecs
    skiprate = winlength // 4
    MIN_SNR = -10
    MAX_SNR = 35

    # For each frame, calculate SSNR
    num_frames = int(clean_length / skiprate - (winlength/skiprate))
    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    segmental_snr = []

    for frame_count in range(int(num_frames)):
        # (1) get the frames for the test and ref speech.
        # Apply Hanning Window
        clean_frame = clean_speech[start:start+winlength]
        processed_frame = processed_speech[start:start+winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Compute Segmental SNR
        signal_energy = np.sum(clean_frame ** 2)
        noise_energy = np.sum((clean_frame - processed_frame) ** 2)
        segmental_snr.append(10 * np.log10(signal_energy / (noise_energy + eps)+ eps))
        segmental_snr[-1] = max(segmental_snr[-1], MIN_SNR)
        segmental_snr[-1] = min(segmental_snr[-1], MAX_SNR)
        start += int(skiprate)
    return overall_snr, segmental_snr

def CompositeEval(ref_wav, deg_wav, mixture, base_dir, log_all=False):
    # returns [sig, bak, ovl]
    alpha = 0.95
    len_ = min(ref_wav.shape[0], deg_wav.shape[0])
    
    ref_wav = ref_wav[:len_]
    ref_len = ref_wav.shape[0]
    deg_wav = deg_wav[:len_]

    # Compute WSS measure
    wss_dist_vec = wss(ref_wav, deg_wav, 16000)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    wss_dist = np.mean(wss_dist_vec[:int(round(len(wss_dist_vec) * alpha))])

    # Compute LLR measure
    LLR_dist = llr(ref_wav, deg_wav, 16000)
    LLR_dist = sorted(LLR_dist, reverse=False)
    LLRs = LLR_dist
    LLR_len = round(len(LLR_dist) * alpha)
    llr_mean = np.mean(LLRs[:LLR_len])

    # Compute the SSNR
    snr_mean, segsnr_mean = SSNR(ref_wav, deg_wav, 16000)
    segSNR = np.mean(segsnr_mean)

    # Compute the PESQ
    pesq_raw = PESQ(ref_wav, deg_wav, mixture)
    nisqa_raw = NISQA(deg_wav, base_dir)
    stoi_raw = STOI_calculate(ref_wav, deg_wav)

    def trim_mos(val):
        return min(max(val, 1), 5)

    Csig = 3.093 - 1.029 * llr_mean + 0.603 * pesq_raw - 0.009 * wss_dist
    Csig = trim_mos(Csig)
    Cbak = 1.634 + 0.478 * pesq_raw - 0.007 * wss_dist + 0.063 * segSNR
    Cbak = trim_mos(Cbak)
    Covl = 1.594 + 0.805 * pesq_raw - 0.512 * llr_mean - 0.007 * wss_dist
    Covl = trim_mos(Covl)
    if log_all:
        return Csig, Cbak, Covl, pesq_raw, segSNR, nisqa_raw, stoi_raw
    else:
        return Csig, Cbak, Covl

def wss(ref_wav, deg_wav, srate):
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]

    assert clean_length == processed_length, clean_length

    winlength = round(30 * srate / 1000.) # 240 wlen in samples
    skiprate = np.floor(winlength / 4)
    max_freq = srate / 2
    num_crit = 25 # num of critical bands

    USE_FFT_SPECTRUM = 1
    n_fft = int(2 ** np.ceil(np.log(2*winlength)/np.log(2)))
    n_fftby2 = int(n_fft / 2)
    Kmax = 20
    Klocmax = 1

    # Critical band filter definitions (Center frequency and BW in Hz)

    cent_freq = [50., 120, 190, 260, 330, 400, 470, 540, 617.372,
                 703.378, 798.717, 904.128, 1020.38, 1148.30, 
                 1288.72, 1442.54, 1610.70, 1794.16, 1993.93, 
                 2211.08, 2446.71, 2701.97, 2978.04, 3276.17,
                 3597.63]
    bandwidth = [70., 70, 70, 70, 70, 70, 70, 77.3724, 86.0056,
                 95.3398, 105.411, 116.256, 127.914, 140.423, 
                 153.823, 168.154, 183.457, 199.776, 217.153, 
                 235.631, 255.255, 276.072, 298.126, 321.465,
                 346.136]

    bw_min = bandwidth[0] # min critical bandwidth

    # set up critical band filters. Note here that Gaussianly shaped filters
    # are used. Also, the sum of the filter weights are equivalent for each
    # critical band filter. Filter less than -30 dB and set to zero.

    min_factor = np.exp(-30. / (2 * 2.303)) # -30 dB point of filter

    crit_filter = np.zeros((num_crit, n_fftby2))
    all_f0 = []
    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * (n_fftby2)
        all_f0.append(np.floor(f0))
        bw = (bandwidth[i] / max_freq) * (n_fftby2)
        norm_factor = np.log(bw_min) - np.log(bandwidth[i])
        j = list(range(n_fftby2))
        crit_filter[i, :] = np.exp(-11 * (((j - np.floor(f0)) / bw) ** 2) + \
                                   norm_factor)
        crit_filter[i, :] = crit_filter[i, :] * (crit_filter[i, :] > \
                                                 min_factor)
    # For each frame of input speech, compute Weighted Spectral Slope Measure

    # num of frames
    num_frames = int(clean_length / skiprate - (winlength / skiprate))
    start = 0 # starting sample
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    distortion = []

    for frame_count in range(num_frames):

        # (1) Get the Frames for the test and reference speeech.
        # Multiply by Hanning window.
        clean_frame = clean_speech[start:start+winlength]
        processed_frame = processed_speech[start:start+winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Compuet Power Spectrum of clean and processed

        clean_spec = (np.abs(np.fft.fft(clean_frame, n_fft)) ** 2)
        processed_spec = (np.abs(np.fft.fft(processed_frame, n_fft)) ** 2)
        clean_energy = [None] * num_crit
        processed_energy = [None] * num_crit
        # (3) Compute Filterbank output energies (in dB)
        for i in range(num_crit):
            clean_energy[i] = np.sum(clean_spec[:n_fftby2] * \
                                     crit_filter[i, :])
            processed_energy[i] = np.sum(processed_spec[:n_fftby2] * \
                                         crit_filter[i, :])
        clean_energy = np.array(clean_energy).reshape(-1, 1)
        eps = np.ones((clean_energy.shape[0], 1)) * 1e-10
        clean_energy = np.concatenate((clean_energy, eps), axis=1)
        clean_energy = 10 * np.log10(np.max(clean_energy, axis=1))
        processed_energy = np.array(processed_energy).reshape(-1, 1)
        processed_energy = np.concatenate((processed_energy, eps), axis=1)
        processed_energy = 10 * np.log10(np.max(processed_energy, axis=1))
        # (4) Compute Spectral Shape (dB[i+1] - dB[i])

        clean_slope = clean_energy[1:num_crit] - clean_energy[:num_crit-1]
        processed_slope = processed_energy[1:num_crit] - \
                processed_energy[:num_crit-1]
        # (5) Find the nearest peak locations in the spectra to each
        # critical band. If the slope is negative, we search
        # to the left. If positive, we search to the right.
        clean_loc_peak = []
        processed_loc_peak = []
        for i in range(num_crit - 1):
            if clean_slope[i] > 0:
                # search to the right
                n = i
                while n < num_crit - 1 and clean_slope[n] > 0:
                    n += 1
                clean_loc_peak.append(clean_energy[n - 1])
            else:
                # search to the left
                n = i
                while n >= 0 and clean_slope[n] <= 0:
                    n -= 1
                clean_loc_peak.append(clean_energy[n + 1])
            # find the peaks in the processed speech signal
            if processed_slope[i] > 0:
                n = i
                while n < num_crit - 1 and processed_slope[n] > 0:
                    n += 1
                processed_loc_peak.append(processed_energy[n - 1])
            else:
                n = i
                while n >= 0 and processed_slope[n] <= 0:
                    n -= 1
                processed_loc_peak.append(processed_energy[n + 1])
        # (6) Compuet the WSS Measure for this frame. This includes
        # determination of the weighting functino
        dBMax_clean = max(clean_energy)
        dBMax_processed = max(processed_energy)
        # The weights are calculated by averaging individual
        # weighting factors from the clean and processed frame.
        # These weights W_clean and W_processed should range
        # from 0 to 1 and place more emphasis on spectral 
        # peaks and less emphasis on slope differences in spectral
        # valleys.  This procedure is described on page 1280 of
        # Klatt's 1982 ICASSP paper.
        clean_loc_peak = np.array(clean_loc_peak)
        processed_loc_peak = np.array(processed_loc_peak)
        Wmax_clean = Kmax / (Kmax + dBMax_clean - clean_energy[:num_crit-1])
        Wlocmax_clean = Klocmax / (Klocmax + clean_loc_peak - \
                                   clean_energy[:num_crit-1])
        W_clean = Wmax_clean * Wlocmax_clean
        Wmax_processed = Kmax / (Kmax + dBMax_processed - \
                                processed_energy[:num_crit-1])
        Wlocmax_processed = Klocmax / (Klocmax + processed_loc_peak - \
                                      processed_energy[:num_crit-1])
        W_processed = Wmax_processed * Wlocmax_processed
        W = (W_clean + W_processed) / 2
        distortion.append(np.sum(W * (clean_slope[:num_crit - 1] - \
                                     processed_slope[:num_crit - 1]) ** 2))

        # this normalization is not part of Klatt's paper, but helps
        # to normalize the meaasure. Here we scale the measure by the sum of the
        # weights
        distortion[frame_count] = distortion[frame_count] / np.sum(W)
        start += int(skiprate)
    return distortion

def llr(ref_wav, deg_wav, srate):
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]

    assert clean_length == processed_length, clean_length

    winlength = round(30 * srate / 1000.) # 240 wlen in samples
    skiprate = np.floor(winlength / 4)
    if srate < 10000:
        # LPC analysis order
        P = 10
    else:
        P = 16

    # For each frame of input speech, calculate the Log Likelihood Ratio

    num_frames = int(clean_length / skiprate - (winlength / skiprate))
    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    distortion = []

    for frame_count in range(num_frames):

        # (1) Get the Frames for the test and reference speeech.
        # Multiply by Hanning window.
        clean_frame = clean_speech[start:start+winlength]
        processed_frame = processed_speech[start:start+winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window
        # (2) Get the autocorrelation logs and LPC params used
        # to compute the LLR measure
        R_clean, Ref_clean, A_clean = lpcoeff(clean_frame, P)
        R_processed, Ref_processed, A_processed = lpcoeff(processed_frame, P)
        A_clean = A_clean[None, :]
        A_processed = A_processed[None, :]
        #print('A_clean shape: ', A_clean.shape)
        #print('toe(R_clean) shape: ', toeplitz(R_clean).shape)
        #print('A_clean: ', A_clean)
        #print('A_processed: ', A_processed)
        #print('toe(R_clean): ', toeplitz(R_clean))
        # (3) Compute the LLR measure
        numerator = A_processed.dot(toeplitz(R_clean)).dot(A_processed.T)
        #print('num_1: {}'.format(A_processed.dot(toeplitz(R_clean))))
        #print('num: ', numerator)
        denominator = A_clean.dot(toeplitz(R_clean)).dot(A_clean.T)
        #print('den: ', denominator)
        #log_ = np.log(max(numerator / denominator, 10e-20))
        #print('R_clean: ', R_clean)
        #print('num: ', numerator)
        #print('den: ', denominator)
        #raise NotImplementedError
        log_ = np.log(max(numerator / denominator, 10e-20))
        #print('np.log({}/{}) = {}'.format(numerator, denominator, log_))
        distortion.append(np.squeeze(log_))
        start += int(skiprate)
    return np.array(distortion)

#@nb.jit('UniTuple(float32[:], 3)(float32[:])')#,nopython=True)
def lpcoeff(speech_frame, model_order):
    
    # (1) Compute Autocor lags
    # max?
    winlength = speech_frame.shape[0]
    R = []
    #R = [0] * (model_order + 1)
    for k in range(model_order + 1):
        first = speech_frame[:(winlength - k)]
        second = speech_frame[k:winlength]
        #raise NotImplementedError
        R.append(np.sum(first * second))
        #R[k] = np.sum( first * second)
    # (2) Lev-Durbin
    a = np.ones((model_order,))
    E = np.zeros((model_order + 1,))
    rcoeff = np.zeros((model_order,))
    E[0] = R[0]
    for i in range(model_order):
        #print('-' * 40)
        #print('i: ', i)
        if i == 0:
            sum_term = 0
        else:
            a_past = a[:i]
            #print('R[i:0:-1] = ', R[i:0:-1])
            #print('a_past = ', a_past)
            sum_term = np.sum(a_past * np.array(R[i:0:-1]))
            #print('a_past size: ', a_past.shape)
        #print('sum_term = {:.6f}'.format(sum_term))
        #print('E[i] =  {}'.format(E[i]))
        #print('R[i+1] = ', R[i+1])
        rcoeff[i] = (R[i+1] - sum_term)/E[i]
        #print('len(a) = ', len(a))
        #print('len(rcoeff) = ', len(rcoeff))
        #print('a[{}]={}'.format(i, a[i]))
        #print('rcoeff[{}]={}'.format(i, rcoeff[i]))
        a[i] = rcoeff[i]
        if i > 0:
            #print('a: ', a)
            #print('a_past: ', a_past)
            #print('a_past[:i] ', a_past[:i])
            #print('a_past[::-1] ', a_past[::-1])
            a[:i] = a_past[:i] - rcoeff[i] * a_past[::-1]
        E[i+1] = (1-rcoeff[i]*rcoeff[i])*E[i]
        #print('E[i+1]= ', E[i+1])
    acorr = np.array(R, dtype=np.float32)
    refcoeff = np.array(rcoeff, dtype=np.float32)
    a = a * -1
    lpparams = np.array([1] + list(a), dtype=np.float32)
    acorr =np.array(acorr, dtype=np.float32)
    refcoeff = np.array(refcoeff, dtype=np.float32)
    lpparams = np.array(lpparams, dtype=np.float32)
    #print('acorr shape: ', acorr.shape)
    #print('refcoeff shape: ', refcoeff.shape)
    #print('lpparams shape: ', lpparams.shape)
    return acorr, refcoeff, lpparams
        
class STOI(nn.Module):
    def __init__(self):
        super(STOI, self).__init__()
        self.fs = 16000
        self.num_bands = 15
        self.center_freq = 150
        self.min_energy = 40
        self.fft_size = 512
        self.fft_in_frame_size = 256
        self.hop = 128
        self.num_frames = 30
        self.beta =  1 + 10**(15 / 20)
        self.fft_pad = (self.fft_size - self.fft_in_frame_size) // 2

        scale = self.fft_size / self.hop
        window = np.hanning(self.fft_in_frame_size)
        zero_pad = np.zeros(self.fft_pad)
        window = np.concatenate([zero_pad, window, zero_pad])
        fft = np.fft.fft(np.eye(self.fft_size))
        self.rows = self.fft_size // 2 + 1
        fft = np.vstack((np.real(fft[:self.rows,:]), np.imag(fft[:self.rows,:])))
        fft = window * fft
        self.fftmat = nn.Parameter(torch.FloatTensor(fft).unsqueeze(1), requires_grad=False)
        self.octmat, _ = self._get_octave_mat(self.fs, self.fft_size,
                                              self.num_bands, self.center_freq)
        self.octmat = nn.Parameter(torch.FloatTensor(self.octmat), requires_grad=False)

    def forward(self, prediction, target, inteference):
        # pred, targ = self._remove_silent_frames(prediction, target)

        # (batch, 1, time) to (batch, fft_size, frames)
        pred_mag, pred_phase = self._stft(prediction)
        targ_mag, targ_phase = self._stft(target)

        # (batch, fft_size, frames) to (batch, frames, fft_size)
        pred_mag = pred_mag.permute(0, 2, 1).contiguous()
        targ_mag = targ_mag.permute(0, 2, 1).contiguous()

        # (batch, frames, fft_size) to (batch, frames, num_bands)
        x = torch.sqrt(F.linear(targ_mag**2, self.octmat))
        y = torch.sqrt(F.linear(pred_mag**2, self.octmat))

        # (batch, frames, num_bands) to (batch, num_bands, frames)
        x = x.permute(0, 2, 1).contiguous()
        y = y.permute(0, 2, 1).contiguous()

        corr = 0
        for i, m in enumerate(range(self.num_frames, x.size()[2])):
            # segment (batch, num_bands, frames) to (batch, num_bands, new_frames)
            x_seg = x[:, :, m - self.num_frames : m]
            y_seg = y[:, :, m - self.num_frames : m]
            alpha = torch.sqrt(torch.sum(x_seg**2, dim=2, keepdim=True) / (torch.sum(y_seg**2, dim=2, keepdim=True) + 1e-7))
            y_prime = torch.min(alpha * y_seg, self.beta * x_seg)
            corr += self._correlation(x_seg, y_prime)

        return -corr / (i + 1)

    def _stft(self, seq):
        seq = seq.unsqueeze(1)
        stft = F.conv1d(seq, self.fftmat, stride=self.hop, padding=self.fft_pad)
        real = stft[:, :self.rows, :]
        imag = stft[:, self.rows:, :]
        mag = torch.sqrt(real**2 + imag**2)
        phase = torch.atan2(imag, real)
        return mag, phase

    def _get_octave_mat(self, fs, nfft, numBands, mn):
        f = np.linspace(0, fs, nfft+1)
        f = f[:int(nfft/2)+1]
        k = np.arange(float(numBands))
        cf = 2**(k/3)*mn;
        fl = np.sqrt((2.**(k/3)*mn) * 2**((k-1.)/3)*mn)
        fr = np.sqrt((2.**(k/3)*mn) * 2**((k+1.)/3)*mn)
        A = np.zeros((numBands, len(f)) )

        for i in range(len(cf)) :
            b = np.argmin((f-fl[i])**2)
            fl[i] = f[b]
            fl_ii = b

            b = np.argmin((f-fr[i])**2)
            fr[i] = f[b]
            fr_ii = b
            A[i, np.arange(fl_ii,fr_ii)] = 1

        rnk = np.sum(A, axis=1)
        numBands = np.where((rnk[1:] >= rnk[:-1]) & (rnk[1:] != 0))[-1][-1]+1
        A = A[:numBands+1,:];
        cf = cf[:numBands+1];
        return A, cf

    def _remove_silent_frames(self, x, y):
        pass

    def _correlation(self, x, y):
        '''
        Input shape is (batch_size, bands, time dimension)
        '''
        xn = x - torch.mean(x, dim=2, keepdim=True)
        xn /= torch.sqrt(torch.sum(xn**2, dim=2, keepdim=True))
        yn = y - torch.mean(y, dim=2, keepdim=True)
        yn /= torch.sqrt(torch.sum(yn**2, dim=2, keepdim=True))
        r = torch.mean(torch.sum(xn * yn, dim=2))
        return r
