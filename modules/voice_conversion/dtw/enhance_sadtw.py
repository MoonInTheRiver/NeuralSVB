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
        # window_boundarys = [[-64, -32], [-32, -16], [-16, -8], [-8, 0], [0, 8], [8, 16], [16, 32], [32, 64]]
        window_boundarys = [[-64, -48], [-48, -32], [-32, -16], [-16, 0], [0, 16], [16, 32], [32, 48], [48, 64]]
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
                if w_idx in [0, 7]:
                    tan_i *= 0.5
                elif w_idx in [1, 6]:
                    tan_i *= 0.75
                elif w_idx in [2, 5]:
                    tan_i *= 0.9
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


def cal_hist_dist(hist_a, hist_b, src, tgt):
    # a: [T1, M]
    # b: [T2, M]
    cartesian_minus = hist_b.unsqueeze(0) - hist_a.unsqueeze(1)  # [T1, T2, M]
    cartesian_add = hist_b.unsqueeze(0) + hist_a.unsqueeze(1)    # [T1, T2, M]
    dist = 0.5 * (cartesian_minus ** 2) / (cartesian_add + 0.00000001)  # [T1, T2, M]
    dist = dist.sum(dim=-1)  # [T1, T2]

    # src_uv = (src != 0).float().unsqueeze(1)  # [T1, 1]  amateur 不是uv的地方 is 1.
    # tgt_uv = (tgt == 0).float().unsqueeze(0)  # [1, T2]  prof 是uv的地方 is 1.
    # uv_matrice = src_uv.matmul(tgt_uv)  # [T1, T2]    (amateur 不是uv的地方) && (prof 是uv的地方) is 1
    # uv_mask = uv_matrice * 2  #  (amateur 不是uv的地方) && (prof 是uv的地方) is 2, otherwise 0
    #
    # dist = dist + uv_mask  # dist的最大值是1（被norm了) 加上这个矩阵之后，invalid的地方就会大于一，不会被对齐
    return dist


def cal_shape_dist(src, tgt, src_len, tgt_len):
    src_hist = torch.tensor(cal_hist_of_f0(src, normalize_hist=True))
    tgt_hist = torch.tensor(cal_hist_of_f0(tgt, normalize_hist=True, scale_factor=tgt_len/src_len))
    return cal_hist_dist(src_hist, tgt_hist, src, tgt).unsqueeze(0)  # [1, S, T]


## here is API for one sample
def EHSADTW(src, tgt, input):
    # src: [S, H]
    # tgt: [T, H]
    dists = cal_shape_dist(src, tgt, src.shape[0], tgt.shape[0])  # [1, S, T]
    costs = dists.squeeze(0)  # [S, T]
    alignment = align_from_distances(costs.T.cpu().detach().numpy())
    output = input[alignment]
    return output, alignment