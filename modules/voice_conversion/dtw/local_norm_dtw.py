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

def get_local_context(input_f, max_window=32, scale_factor=1.):
    # input_f: [S, 1], support numpy array or torch tensor
    # return hist: [S, max_window * 2], list of list
    T = input_f.shape[0]
    # max_window = int(max_window * scale_factor)
    derivative = [[0 for _ in range(max_window * 2)] for _ in range(T)]

    for t in range(T):   # travel the time series
        for feat_idx in range(-max_window, max_window):
            if t + feat_idx < 0 or t + feat_idx >= T:
                value = 0
            else:
                value = input_f[t+feat_idx]
            derivative[t][feat_idx+max_window] = value
    return derivative

def cal_localnorm_dist(src, tgt, src_len, tgt_len):
    local_src = torch.tensor(get_local_context(src))
    local_tgt = torch.tensor(get_local_context(tgt, scale_factor=tgt_len / src_len))

    local_norm_src = (local_src - local_src.mean(-1).unsqueeze(-1)) #/ (local_src.std(-1).unsqueeze(-1) +0.00000001) # [T1, 32]
    local_norm_tgt = (local_tgt - local_tgt.mean(-1).unsqueeze(-1)) #/ (local_tgt.std(-1).unsqueeze(-1) +0.00000001) # [T2, 32]

    dists = torch.cdist(local_norm_src[None, :, :], local_norm_tgt[None, :, :])  # [1, T1, T2]
    return dists


def DTW_align(srcs, tgts, inputs, srcs_lens=None, tgt_lens=None):
    # srcs: [B, S, M]
    # tgts: [B, T, M]
    # inputs: [B, S, H]
    # outputs: [B, T, H]
    # alignment: [T]  range from [0, S]
    outputs = []
    for src, tgt, input, src_len, tgt_len in zip(srcs, tgts, inputs, srcs_lens, tgt_lens):
        dists = cal_localnorm_dist(src, tgt, src_len.item(), tgt_len.item())  # [1, S, T]
        costs = dists.squeeze(0)   # [S, T]
        alignment = align_from_distances(costs.T.cpu().detach().numpy())
        output = input[alignment]
        outputs.append(output)
    return torch.stack(outputs, dim=0)


## here is API for one sample
def LoNDTW(src, tgt, input):
    # src: [S, H]
    # tgt: [T, H]
    dists = cal_localnorm_dist(src, tgt, src.shape[0], tgt.shape[0])  # [1, S, T]
    costs = dists.squeeze(0)  # [S, T]
    alignment = align_from_distances(costs.T.cpu().detach().numpy())
    output = input[alignment]
    return output, alignment


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
    aligned_mel, alignment = LoNDTW(amateur_f0_denorm, prof_f0_denorm, amateur_mel)
    aligned_f0_denorm, alignment = LoNDTW(amateur_f0_denorm, prof_f0_denorm, amateur_f0_denorm)
    cat_spec = torch.cat([amateur_mel_padded, aligned_mel, prof_mel], dim=-1)
    spec_to_figure(cat_spec, name=f'f0_denorm_mel_64_B.png')
    # f0 align f0
    f0_to_figure(f0_src=amateur_f0_denorm, f0_aligned=aligned_f0_denorm, f0_prof=prof_f0_denorm,
                 name=f'f0_denorm_f0_64_B.png')
    amateur_mel2ph = sample['mel2ph']
    prof_mel2ph = sample['prof_mel2ph']
    aligned_mel2ph = amateur_mel2ph[alignment]
    acc = (prof_mel2ph == aligned_mel2ph).sum().cpu().numpy() / (
            prof_mel2ph != 0).sum().cpu().numpy()
    print(acc)
    exit()

    sample = train_ds.collater([train_ds[0]])#, train_ds[1], train_ds[2],train_ds[3],train_ds[4],train_ds[5],train_ds[6],train_ds[7],train_ds[8],train_ds[9]])

    amateur_mel2ph = sample['mel2ph']
    prof_mel2ph = sample['prof_mel2ph']

    amateur_f0 = sample['f0']
    prof_f0 = sample['prof_f0']

    amateur_uv = sample['uv']
    amateur_padding = amateur_mel2ph == 0
    prof_uv = sample['prof_uv']
    prof_padding = prof_mel2ph == 0
    amateur_f0_denorm = denorm_f0(amateur_f0, amateur_uv, hparams, pitch_padding=amateur_padding)
    prof_f0_denorm = denorm_f0(prof_f0, prof_uv, hparams, pitch_padding=prof_padding)

    amateur_mel = sample['mels']
    prof_mel = sample['prof_mels']
    pad_num = max(prof_mel.shape[1] - amateur_mel.shape[1], 0)
    amateur_mel_padded = F.pad(amateur_mel, [0, 0, 0, pad_num])[:, :prof_mel.shape[1], :]

    amateur_lens = (sample['mel2ph'] != 0).sum(-1)
    prof_lens = (sample['prof_mel2ph'] != 0).sum(-1)
    aligned_mel = DTW_align(amateur_f0_denorm, prof_f0_denorm, amateur_mel, srcs_lens=amateur_lens, tgt_lens=prof_lens)
    aligned_f0_denorm = DTW_align(amateur_f0_denorm, prof_f0_denorm, amateur_f0_denorm, srcs_lens=amateur_lens, tgt_lens=prof_lens)

    # align mel2ph
    aligned_mel2ph = DTW_align(amateur_f0_denorm, prof_f0_denorm, amateur_mel2ph, srcs_lens=amateur_lens, tgt_lens=prof_lens)

    cat_spec = torch.cat([amateur_mel_padded, aligned_mel, prof_mel], dim=-1)

    for idx in range(amateur_f0.shape[0]):  # [B]
        spec_to_figure(cat_spec[idx], name=f'f0_denorm_mel_64_{idx}.png')
        # f0 align f0
        f0_to_figure(f0_src=amateur_f0_denorm[idx], f0_aligned=aligned_f0_denorm[idx], f0_prof=prof_f0_denorm[idx],
                 name=f'f0_denorm_f0_64_{idx}.png')

        acc = (prof_mel2ph[idx] == aligned_mel2ph[idx]).sum().cpu().numpy() / (prof_mel2ph[idx] != 0).sum().cpu().numpy()
        print(acc)
# python modules/voice_conversion/dtw/local_norm_dtw.py --config egs/datasets/audio/PopBuTFy/svc_ppg.yaml