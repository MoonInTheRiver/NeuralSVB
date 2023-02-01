import os
os.environ["OMP_NUM_THREADS"] = "1"

import glob
from tqdm import tqdm
from multiprocessing.pool import Pool
import numpy as np
import torch.nn.functional as F
import torch

# utils from dtw
from modules.voice_conversion.dtw.shape_aware_dtw import SADTW, f0_to_figure, spec_to_figure
from modules.voice_conversion.dtw.local_norm_dtw import LoNDTW
from modules.voice_conversion.dtw.naive_dtw import NaiveDTW, ZMNaiveDTW, NNaiveDTW
from modules.voice_conversion.dtw.naive_interpo import NInterpo
from modules.voice_conversion.dtw.enhance_sadtw import EHSADTW

# utils from ns
from utils.pitch_utils import denorm_f0
from tasks.singing.neural_svb_task import FastSingingDataset
from utils.hparams import hparams, set_hparams

thresh_hold = 0.3
prefix = 'a2p'
choosed_func = 'EHSADTW'
# choosed_func = 'SADTW'
# choosed_func = 'LoNDTW'
# choosed_func = 'ZMNaiveDTW'
# choosed_func = 'NNaiveDTW'
# choosed_func = 'NaiveDTW'

align_funcs = {
    'SADTW': SADTW,
    'LoNDTW': LoNDTW,
    'ZMNaiveDTW': ZMNaiveDTW,
    'NNaiveDTW': NNaiveDTW,
    'EHSADTW': EHSADTW,
    'NaiveDTW': NaiveDTW  # calculate on clean mel to test upper bound
}

def job(sample, processed_path, align_func=None, save_res=False, show_f0=True):
    item_name = sample['item_name']
    if prefix == 'a2p':
        amateur_f0 = sample['f0']
        amateur_uv = sample['uv']
        amateur_mel2ph = sample['mel2ph']
        amateur_mel = sample['mel']
        amateur_padding = amateur_mel2ph == 0
        amateur_f0_denorm = denorm_f0(amateur_f0, amateur_uv, hparams, pitch_padding=amateur_padding)

        prof_f0 = sample['prof_f0']
        prof_uv = sample['prof_uv']
        prof_mel2ph = sample['prof_mel2ph']
        prof_mel = sample['prof_mel']
        prof_padding = prof_mel2ph == 0
        prof_f0_denorm = denorm_f0(prof_f0, prof_uv, hparams, pitch_padding=prof_padding)
    elif prefix == 'p2a':
        amateur_f0 = sample['prof_f0']
        amateur_uv = sample['prof_uv']
        amateur_mel2ph = sample['prof_mel2ph']
        amateur_mel = sample['prof_mel']
        amateur_padding = amateur_mel2ph == 0
        amateur_f0_denorm = denorm_f0(amateur_f0, amateur_uv, hparams, pitch_padding=amateur_padding)

        prof_f0 = sample['f0']
        prof_uv = sample['uv']
        prof_mel2ph = sample['mel2ph']
        prof_mel = sample['mel']
        prof_padding = prof_mel2ph == 0
        prof_f0_denorm = denorm_f0(prof_f0, prof_uv, hparams, pitch_padding=prof_padding)
    else:
        print('Bad case of prefix.')
        exit(1)

    aligned_f0_denorm, alignment = align_func(amateur_f0_denorm, prof_f0_denorm, amateur_f0_denorm)

    aligned_mel2ph = amateur_mel2ph[alignment]
    acc = (prof_mel2ph == aligned_mel2ph).sum().cpu().numpy() / (
            prof_mel2ph != 0).sum().cpu().numpy()

    if show_f0 and acc < thresh_hold:
        pad_num = max(prof_mel.shape[0] - amateur_mel.shape[0], 0)
        amateur_mel_padded = F.pad(amateur_mel, [0, 0, 0, pad_num])[:prof_mel.shape[0], :]
        aligned_mel = amateur_mel[alignment] #, alignment = align_func(amateur_f0_denorm, prof_f0_denorm, amateur_mel)
        # aligned_f0_denorm, alignment = align_func(amateur_f0_denorm, prof_f0_denorm, amateur_f0_denorm)
        cat_spec = torch.cat([amateur_mel_padded, aligned_mel, prof_mel], dim=-1)
        spec_to_figure(cat_spec, name=f'mel_{item_name}_{acc}.png', dir=f'tmp_{prefix}_{choosed_func}')
        f0_to_figure(f0_src=amateur_f0_denorm, f0_aligned=aligned_f0_denorm, f0_prof=prof_f0_denorm, name=f'f0_{item_name}_{acc}.png', dir=f'tmp_{prefix}_{choosed_func}')

    if save_res:
        np.save(os.path.join(processed_path, f'{item_name}_alignment.npy'), alignment)
    return [acc, item_name]


class MyMultiprocessor:
    def __init__(self):
        super().__init__()
        set_hparams()
        self.processed_path = os.path.join(hparams['processed_data_dir'], f'{prefix}_alignments_by_{choosed_func}')
        self.train_ds = FastSingingDataset('train')  # for training set.
        self.test_ds = FastSingingDataset('test')   # for testing set.
        os.makedirs(self.processed_path, exist_ok=True)

    def multi_processor(self, ds):
        saving_result_pool = Pool(20)
        saving_results_futures = []
        for sample in ds:
            saving_results_futures.append([
                saving_result_pool.apply_async(job, args=(sample, self.processed_path, align_funcs[choosed_func], False, True))])
        saving_result_pool.close()

        acc_dict = {}
        for f_id, future in enumerate(tqdm(saving_results_futures)):
            try:
                res = future[0].get()
                acc_dict[res[1]] = res[0]
                saving_results_futures[f_id] = None
            except Exception:
                print('Sample: ', res[1], 'failed!.')
                continue
        saving_result_pool.join()

        # valid_accs = [acc for acc in acc_dict.values() if acc > thresh_hold]
        valid_accs = list(acc_dict.values())
        print('avg', sum(valid_accs) / len(valid_accs), 'max', max(valid_accs), 'min', min(valid_accs))
        # import matplotlib.pyplot as plt
        # plt.hist(acc_dict.values(), bins=20)
        # plt.savefig(f'{choosed_func}_hist.png')
        # if os.path.exists(f'{self.processed_path}/acc_dict.npy'):
        #     a = np.load(f'{self.processed_path}/acc_dict.npy', allow_pickle=True).item()
        #     a.update(acc_dict)
        #     acc_dict = a
        # np.save(f'{self.processed_path}/acc_dict.npy', acc_dict)

if __name__ == '__main__':
    set_hparams()
    pitch_alignmentor = MyMultiprocessor()
    pitch_alignmentor.multi_processor(pitch_alignmentor.train_ds)
    pitch_alignmentor.multi_processor(pitch_alignmentor.test_ds)
    print('Process done!')