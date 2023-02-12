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


def cal_hist_of_f0(input_f, max_window=64, normalize_hist=False, scale_factor=1.):
    # input_f: [S, 1], support numpy array or torch tensor
    # return hist: [S, region_number * window_number], list of list
    T = input_f.shape[0]
    if max_window == 128:
        window_boundarys = [[-128, -64], [-64, -32], [-32, -16], [-16, -8], [-8, 0], [0, 8], [8, 16], [16, 32], [32, 64], [64, 128]]
    elif max_window == 64:
        window_boundarys = [[-64, -32], [-32, -16], [-16, -8], [-8, 0], [0, 8], [8, 16], [16, 32], [32, 64]]
    elif max_window == 32:
        window_boundarys = [[-32, -16], [-16, -8], [-8, 0], [0, 8], [8, 16], [16, 32]]
    else:  # case for code testing
        window_boundarys = [[-6, -4], [-4, -2], [-2, 0], [1, 2], [2, 4], [4, 6]]

    region_number = 6   # the number of regions in a window

    derivative = [[0 for _ in range(region_number * len(window_boundarys))] for _ in range(T)]

    for t in range(T):   # travel the time series
        total_t = 0
        for w_idx in range(len(window_boundarys)):
            relative_left, relative_right = window_boundarys[w_idx]
            relative_left = int(relative_left * scale_factor)
            relative_right = int(relative_right * scale_factor)
            if relative_left == 0:
                relative_left = 1
            left_boundary = min(max(0, relative_left + t), T)
            right_boundary = min(max(0, relative_right + t), T)
            for i in range(left_boundary, right_boundary):
                assert (i - t) != 0, (left_boundary, right_boundary)
                tan_i = (input_f[i] - input_f[t]) / (i-t)
                region_idx = None
                if 0 <= abs(tan_i) < 0.57735:
                    if (input_f[i] - input_f[t]) >= 0:  # above x_axis
                        region_idx = 2
                    else:
                        region_idx = 3
                elif 0.57735 <= abs(tan_i) < 1.73205:
                    if (input_f[i] - input_f[t]) >= 0:  # above x_axis
                        region_idx = 1
                    else:
                        region_idx = 4
                elif 1.73205 <= abs(tan_i):
                    if (input_f[i] - input_f[t]) >= 0:  # above x_axis
                        region_idx = 0
                    else:
                        region_idx = 5
                else:
                    print('invalid case!')
                    exit(1)
                derivative[t][w_idx * region_number + region_idx] += 1
                total_t += 1
        if normalize_hist:
            for _idx in range(region_number * len(window_boundarys)):
                derivative[t][_idx] = derivative[t][_idx] / total_t

    return derivative


def cal_hist_dist(hist_a, hist_b):
    # a: [T1, M]
    # b: [T2, M]
    cartesian_minus = hist_b.unsqueeze(0) - hist_a.unsqueeze(1)  # [T1, T2, M]
    cartesian_add = hist_b.unsqueeze(0) + hist_a.unsqueeze(1)    # [T1, T2, M]
    dist = 0.5 * (cartesian_minus ** 2) / (cartesian_add + 0.00000001)  # [T1, T2, M]
    return dist.sum(dim=-1)  # [T1, T2]


def cal_shape_dist(src, tgt, src_len, tgt_len):
    src = torch.tensor(cal_hist_of_f0(src, normalize_hist=True))
    tgt = torch.tensor(cal_hist_of_f0(tgt, normalize_hist=True, scale_factor=tgt_len/src_len))
    return cal_hist_dist(src, tgt).unsqueeze(0)  # [1, S, T]

## here is API for one batch
def DTW_align(srcs, tgts, inputs, srcs_lens=None, tgt_lens=None):
    # srcs: [B, S, H]
    # tgts: [B, T, H]
    # inputs: [B, S, H]
    # outputs: [B, T, H]
    # alignment: [T]  range from [0, S]
    outputs = []
    for src, tgt, input, src_len, tgt_len in zip(srcs, tgts, inputs, srcs_lens, tgt_lens):
        dists = cal_shape_dist(src, tgt, src_len.item(), tgt_len.item())  # [1, S, T]
        costs = dists.squeeze(0)   # [S, T]
        alignment = align_from_distances(costs.T.cpu().detach().numpy())
        output = input[alignment]
        outputs.append(output)
    return torch.stack(outputs, dim=0)


## here is API for one sample
def SADTW(src, tgt, input):
    # src: [S, H]
    # tgt: [T, H]
    dists = cal_shape_dist(src, tgt, src.shape[0], tgt.shape[0])  # [1, S, T]
    costs = dists.squeeze(0)  # [S, T]
    alignment = align_from_distances(costs.T.cpu().detach().numpy())
    output = input[alignment]
    return output, alignment


# code for visualization
def spec_to_figure(spec, vmin=None, vmax=None, name='', dir='tmp'):
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    fig = plt.figure(figsize=(12, 6))
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    os.makedirs(dir, exist_ok=True)
    plt.savefig(os.path.join(dir, name))
    return fig

def f0_to_figure(f0_src, f0_aligned=None, f0_prof=None, name='f0.png', dir='tmp'):
    fig = plt.figure(figsize=(12, 8))
    f0_src = f0_src #.cpu().numpy()
    f0_src[f0_src == 0] = np.nan
    plt.plot(f0_src, color='r', label='src')
    if f0_aligned is not None:
        f0_aligned = f0_aligned # .cpu().numpy()
        f0_aligned[f0_aligned == 0] = np.nan
        plt.plot(f0_aligned, color='b', label='f0_aligned')
    if f0_prof is not None:
        f0_prof = f0_prof # .cpu().numpy()
        f0_prof[f0_prof == 0] = np.nan
        plt.plot(f0_prof, color='green', label='profession')
    plt.legend()
    os.makedirs(dir, exist_ok=True)
    plt.savefig(os.path.join(dir, name))
    return fig

if __name__ == '__main__':

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
    aligned_mel, alignment = SADTW(amateur_f0_denorm, prof_f0_denorm, amateur_mel)
    aligned_f0_denorm, alignment = SADTW(amateur_f0_denorm, prof_f0_denorm, amateur_f0_denorm)
    cat_spec = torch.cat([amateur_mel_padded, aligned_mel, prof_mel], dim=-1)
    spec_to_figure(cat_spec, name=f'f0_denorm_mel_64_A.png')
    # f0 align f0
    f0_to_figure(f0_src=amateur_f0_denorm, f0_aligned=aligned_f0_denorm, f0_prof=prof_f0_denorm,
                 name=f'f0_denorm_f0_64_A.png')
    amateur_mel2ph = sample['mel2ph']
    prof_mel2ph = sample['prof_mel2ph']
    aligned_mel2ph = amateur_mel2ph[alignment]
    acc = (prof_mel2ph == aligned_mel2ph).sum().cpu().numpy() / (
            prof_mel2ph != 0).sum().cpu().numpy()
    print(acc)
    exit()

    sample = train_ds.collater([train_ds[0]]) #, train_ds[1], train_ds[2],train_ds[3],train_ds[4],train_ds[5],train_ds[6],train_ds[7],train_ds[8],train_ds[9]])

    amateur_f0 = sample['f0']
    prof_f0 = sample['prof_f0']

    amateur_uv = sample['uv']
    amateur_padding = sample['mel2ph'] == 0
    prof_uv = sample['prof_uv']
    prof_padding = sample['prof_mel2ph'] == 0
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
    cat_spec = torch.cat([amateur_mel_padded, aligned_mel, prof_mel], dim=-1)

    # align mel2ph
    amateur_mel2ph = sample['mel2ph']
    prof_mel2ph = sample['prof_mel2ph']
    aligned_mel2ph = DTW_align(amateur_f0_denorm, prof_f0_denorm, amateur_mel2ph, srcs_lens=amateur_lens,
                               tgt_lens=prof_lens)

    for idx in range(amateur_f0.shape[0]):  # [B]
        spec_to_figure(cat_spec[idx], name=f'f0_denorm_mel_64_{idx}.png')
        # f0 align f0
        f0_to_figure(f0_src=amateur_f0_denorm[idx], f0_aligned=aligned_f0_denorm[idx], f0_prof=prof_f0_denorm[idx],
                 name=f'f0_denorm_f0_64_{idx}.png')

        acc = (prof_mel2ph[idx] == aligned_mel2ph[idx]).sum().cpu().numpy() / (
                    prof_mel2ph[idx] != 0).sum().cpu().numpy()
        print(acc)

# python modules/voice_conversion/dtw/shape_aware_dtw.py --config egs/datasets/audio/PopBuTFy/svc_ppg.yaml