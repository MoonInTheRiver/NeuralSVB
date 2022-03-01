from utils.indexed_datasets import IndexedDataset
from tqdm import tqdm
from modules.voice_conversion.dtw.align import align_from_distances
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
import time
import os
from multiprocessing import Pool
from utils.pitch_utils import f0_to_coarse, denorm_f0
import numpy as np

from tasks.singing.neural_svb_task import FastSingingDataset
from utils.hparams import hparams, set_hparams

def NInterpo(src, tgt, input, amateur_mel2ph, amateur_mel=None):
    # src: [S]
    # tgt: [T]
    output = F.interpolate(src[None, None, :], size=tgt.shape[-1], mode='nearest').transpose(1, 2)[0]
    if amateur_mel is not None:
        aligned_mel = F.interpolate(amateur_mel[None, :, :].transpose(1, 2), size=tgt.shape[-1], mode='nearest').transpose(1, 2)[0]
    else:
        aligned_mel = None
    aligned_mel2ph = F.interpolate(amateur_mel2ph.float()[None, None, :], size=tgt.shape[-1], mode='nearest').transpose(1, 2)[0].long()
    return output, aligned_mel2ph, aligned_mel

if __name__ == '__main__':
    # code for visualization
    def spec_to_figure(spec, vmin=None, vmax=None, name=''):
        if isinstance(spec, torch.Tensor):
            spec = spec.cpu().numpy()
        fig = plt.figure(figsize=(12, 6))
        plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
        plt.savefig(os.path.join('tmp', name))
        return fig

    def f0_to_figure(f0_src, f0_aligned=None, f0_prof=None, name='f0.png'):
        fig = plt.figure(figsize=(12, 8))
        f0_src = f0_src.cpu().numpy()
        f0_src[f0_src == 0] = np.nan
        plt.plot(f0_src, color='r', label='src')
        if f0_aligned is not None:
            f0_aligned = f0_aligned.cpu().numpy()
            f0_aligned[f0_aligned == 0] = np.nan
            plt.plot(f0_aligned, color='b', label='f0_aligned')
        if f0_prof is not None:
            f0_pred = f0_prof.cpu().numpy()
            f0_prof[f0_prof == 0] = np.nan
            plt.plot(f0_pred, color='green', label='profession')
        plt.legend()
        plt.savefig(os.path.join('tmp', name))
        return fig

    set_hparams()

    train_ds = FastSingingDataset('test')

    # Test One sample case
    sample = train_ds[0]
    amateur_f0 = sample['f0']
    prof_f0 = sample['prof_f0']

    amateur_uv = sample['uv']
    amateur_padding = sample['mel2ph'] == 0
    prof_uv = sample['prof_uv']
    prof_padding = sample['prof_mel2ph'] == 0
    amateur_f0_denorm = denorm_f0(amateur_f0, amateur_uv, hparams, pitch_padding=amateur_padding)
    prof_f0_denorm = denorm_f0(prof_f0, prof_uv, hparams, pitch_padding=prof_padding)
    amateur_mel = sample['mel']
    prof_mel = sample['prof_mel']
    pad_num = max(prof_mel.shape[0] - amateur_mel.shape[0], 0)
    amateur_mel_padded = F.pad(amateur_mel, [0, 0, 0, pad_num])[:prof_mel.shape[0], :]

    amateur_mel2ph = sample['mel2ph']
    prof_mel2ph = sample['prof_mel2ph']
    # interpolate need [B, C, T]
    aligned_f0_denorm, aligned_mel2ph, aligned_mel = NInterpo(amateur_f0, prof_f0, amateur_f0, amateur_mel2ph, amateur_mel=amateur_mel)

    cat_spec = torch.cat([amateur_mel_padded, aligned_mel, prof_mel], dim=-1)
    spec_to_figure(cat_spec, name=f'f0_denorm_mel_D.png')
    # f0 align f0
    f0_to_figure(f0_src=amateur_f0_denorm, f0_aligned=aligned_f0_denorm, f0_prof=prof_f0_denorm,
                 name=f'f0_denorm_f0_D.png')

    acc = (prof_mel2ph == aligned_mel2ph).sum().cpu().numpy() / (
            prof_mel2ph != 0).sum().cpu().numpy()
    print(acc)
    exit()

# python modules/voice_conversion/dtw/naive_interpo.py --config egs/datasets/audio/PopBuTFy/svc_ppg.yaml