from data_gen.singing.binarize import SingingBinarizer, split_train_test_set
from data_gen.tts.base_binarizer import BinarizationError
import re
import os
from utils.multiprocess_utils import chunked_multiprocess_run
import random
import traceback
from resemblyzer import VoiceEncoder
from tqdm import tqdm
from data_gen.tts.data_gen_utils import get_mel2ph, get_pitch, build_phone_encoder, is_sil_phoneme
from utils.hparams import hparams, set_hparams
import numpy as np
from utils.indexed_datasets import IndexedDatasetBuilder
from vocoders.base_vocoder import get_vocoder_cls
import pandas as pd

# utils from dtw
from modules.voice_conversion.dtw.shape_aware_dtw import SADTW, f0_to_figure, spec_to_figure
from modules.voice_conversion.dtw.local_norm_dtw import LoNDTW
from modules.voice_conversion.dtw.naive_dtw import NaiveDTW, ZMNaiveDTW, NNaiveDTW
from modules.voice_conversion.dtw.naive_interpo import NInterpo
from modules.voice_conversion.dtw.enhance_sadtw import EHSADTW


class SaveSpkEmb(SingingBinarizer):
    def load_meta_data(self):
        super().load_meta_data()
        new_item_names = []
        n_utt_ds = {k: 0 for k in hparams['datasets']}
        for item_name in self.item_names:
            if '#singing#' not in item_name:
                continue
            for dataset in hparams['datasets']:
                if len(re.findall(rf'{dataset}', item_name)) > 0:
                    new_item_names.append(item_name)
                    n_utt_ds[dataset] += 1
                    break
        print('n_utt_ds: ', n_utt_ds)
        self.item_names = new_item_names
        self._train_item_names, self._test_item_names = split_train_test_set(self.item_names)

    def process_data(self, prefix):
        spk_emb_dir = hparams['spk_emb_data_dir']
        os.makedirs(spk_emb_dir, exist_ok=True)
        args = []
        voice_encoder = VoiceEncoder().cuda()
        meta_data = list(self.meta_data(prefix))
        for m in meta_data:
            args.append(list(m) + [self.binarization_args])
        num_workers = self.num_workers
        for f_id, (_, item) in enumerate(
                zip(tqdm(meta_data), chunked_multiprocess_run(self.process_item, args, num_workers=num_workers))):
            if item is None:
                continue
            item['spk_embed'] = voice_encoder.embed_utterance(item['wav'])
            np.save(os.path.join(spk_emb_dir, item['item_name'] + '.npy'), item['spk_embed'])

    @classmethod
    def process_item(cls, item_name, wav_fn, spk_id, binarization_args):
        res = {'item_name': item_name, 'wav_fn': wav_fn, 'spk_id': spk_id}
        if binarization_args['with_linear']:
            wav, mel, linear_stft = get_vocoder_cls(hparams).wav2spec(wav_fn, return_linear=True)
            res['linear'] = linear_stft
        else:
            wav, mel = get_vocoder_cls(hparams).wav2spec(wav_fn)
        res.update({'mel': mel, 'wav': wav,
                    'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0],
                    })
        return res


class PopBuTFyENBinarizer(SingingBinarizer):
    def load_meta_data(self):
        super().load_meta_data()
        self.amateur2profwavfn = {}

        new_item_names = []
        n_utt_ds = {k: 0 for k in hparams['datasets']}
        unpaired_num = 0
        for item_name in self.item_names:
            if '#singing#' not in item_name:
                continue
            if 'Professional' in item_name:
                continue
            for dataset in hparams['datasets']:
                if len(re.findall(rf'{dataset}', item_name)) > 0:
                    prof_item = item_name.replace('Amateur', 'Professional')
                    if self.item2wavfn.get(prof_item) is not None and os.path.exists(self.item2wavfn[prof_item]):
                        # check lyric
                        # if self.item2txt[item_name] != self.item2txt[prof_item]:
                        #     print('===> Lyric not match !! : ', item_name, self.item2txt[item_name], self.item2txt[prof_item])
                        #     exit(1)
                        self.amateur2profwavfn[item_name] = self.item2wavfn[prof_item]
                        new_item_names.append(item_name)
                        n_utt_ds[dataset] += 1
                    else:
                        # print('===> Not paired: ', self.item2wavfn[item_name])
                        unpaired_num += 1
        print('n_utt_ds: ', n_utt_ds, 'Unpaired data: ', unpaired_num)
        self.item_names = new_item_names
        self._train_item_names, self._test_item_names = split_train_test_set(self.item_names)

    def meta_data(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            wav_fn = self.item2wavfn[item_name]
            spk_id = self.item_name2spk_id(item_name)
            profwavfn = self.amateur2profwavfn[item_name]
            yield item_name, wav_fn, spk_id, profwavfn

    def process_data(self, prefix):
        data_dir = hparams['binary_data_dir']
        args = []
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        ph_lengths = []
        mel_lengths = []
        f0s = []
        total_sec = 0
        if self.binarization_args['with_spk_embed']:
            voice_encoder = VoiceEncoder().cuda()

        meta_data = list(self.meta_data(prefix))
        for m in meta_data:
            args.append(list(m) + [self.binarization_args])
        num_workers = self.num_workers
        for f_id, (_, item) in enumerate(
                zip(tqdm(meta_data), chunked_multiprocess_run(self.process_item, args, num_workers=num_workers))):
            if item is None:
                continue
            item['spk_embed'] = voice_encoder.embed_utterance(item['wav']) \
                if self.binarization_args['with_spk_embed'] else None
            if not self.binarization_args['with_wav'] and 'wav' in item:
                del item['wav']
                del item['prof_wav']
            builder.add_item(item)
            mel_lengths.append(max(item['len'], item['prof_len']))    # Here changed ! max of two tech
            if 'ph_len' in item:
                ph_lengths.append(item['ph_len'])
            total_sec += item['sec']
            if item.get('f0') is not None:
                f0s.append(item['f0'])
                f0s.append(item['prof_f0'])  # professional f0.         # Here changed ! mean std of two tech
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_lengths.npy', mel_lengths)
        if len(ph_lengths) > 0:
            np.save(f'{data_dir}/{prefix}_ph_lengths.npy', ph_lengths)
        if len(f0s) > 0:
            f0s = np.concatenate(f0s, 0)
            f0s = f0s[f0s != 0]
            np.save(f'{data_dir}/{prefix}_f0s_mean_std.npy', [np.mean(f0s).item(), np.std(f0s).item()])
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    @staticmethod
    def get_pitch(res, prefix=''):
        wav, mel = res[f'{prefix}wav'], res[f'{prefix}mel']
        f0, pitch_coarse = get_pitch(wav, mel, hparams)
        if sum(f0) == 0:
            raise BinarizationError("Empty f0")
        res[f'{prefix}f0'] = f0
        res[f'{prefix}pitch'] = pitch_coarse

    @staticmethod
    def get_pitch_align(res, amateur_f0, prof_f0, item_name, prefix='a2p', choosed_func='EHSADTW'):
        align_funcs = {
            'SADTW': SADTW,
            'LoNDTW': LoNDTW,
            'ZMNaiveDTW': ZMNaiveDTW,
            'NNaiveDTW': NNaiveDTW,
            'EHSADTW': EHSADTW,
            'NaiveDTW': NaiveDTW  # calculate on clean mel to test upper bound
        }
        if prefix == 'a2p':
            aligned_f0, alignment = align_funcs[choosed_func](amateur_f0, prof_f0, amateur_f0)
        else:
            return

        # f0_to_figure(f0_src=amateur_f0, f0_aligned=aligned_f0, f0_prof=prof_f0,
        #              name=f'f0_{item_name}.png', dir=f'tmp_{prefix}_{choosed_func}')

        res[f'{prefix}_f0_alignment'] = alignment

    @classmethod
    def process_item(cls, item_name, wav_fn, spk_id, profwavfn, binarization_args):
        res = {'item_name': item_name, 'wav_fn': wav_fn, 'spk_id': spk_id, 'a2profwavfn': profwavfn,
               }

        wav, mel = get_vocoder_cls(hparams).wav2spec(wav_fn)
        prof_wav, prof_mel = get_vocoder_cls(hparams).wav2spec(profwavfn)

        if hparams.get('max_mel_tech_gap') is not None and abs(mel.shape[0] - prof_mel.shape[0]) > hparams['max_mel_tech_gap']:
            # print('Gap is too large: ', item_name, mel.shape, prof_mel.shape)
            with open(hparams['binary_data_dir'] + '/bad_case.txt', 'a+') as wf:
                wf.write('Gap is too large: ' + item_name + str(mel.shape) + str(prof_mel.shape) + '\n')
            return None
        res.update({'mel': mel, 'wav': wav, 'prof_mel': prof_mel, 'prof_wav': prof_wav,
                    'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0],
                   'prof_sec': len(prof_wav) / hparams['audio_sample_rate'], 'prof_len': prof_mel.shape[0]})
        try:
            if binarization_args['with_f0']:
                cls.get_pitch(res)
                cls.get_pitch(res, prefix='prof_')  # prof wav/mel 2 prof f0/ prof pitch
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        except Exception as e:
            traceback.print_exc()
            print(f"| Skip item. item_name: {item_name}, wav_fn: {wav_fn}")
            return None

        cls.get_pitch_align(res, amateur_f0=res['f0'], prof_f0=res['prof_f0'], item_name=item_name)

        return res

class PopBuTFyENSpkEMBinarizer(PopBuTFyENBinarizer):
    def meta_data(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            wav_fn = self.item2wavfn[item_name]
            spk_id = self.item_name2spk_id(item_name)
            profwavfn = self.amateur2profwavfn[item_name]
            yield item_name, wav_fn, spk_id, profwavfn, self.item_names

    @classmethod
    def process_item(cls, item_name, wav_fn, spk_id, profwavfn, item_names, binarization_args):
        res = super().process_item(item_name, wav_fn, spk_id, profwavfn, binarization_args)
        song_name = item_name[:-re.search(r'_', item_name[::-1]).span()[0]]
        song_pieces = [song for song in item_names if song_name in song]
        random.shuffle(song_pieces)
        select_spk_emb = song_pieces[:hparams['spk_emb_num']]
        multi_spk_emb = []

        try:
            # add itself
            spk_npy_path = os.path.join(hparams['spk_emb_data_dir'], item_name + '.npy')
            multi_spk_emb.append(np.load(spk_npy_path, allow_pickle=True))

            for i in range(hparams['spk_emb_num']):
                if i >= len(select_spk_emb):
                    spk_npy_path = os.path.join(hparams['spk_emb_data_dir'], select_spk_emb[-1] + '.npy')
                else:
                    spk_npy_path = os.path.join(hparams['spk_emb_data_dir'], select_spk_emb[i] + '.npy')
                multi_spk_emb.append(np.load(spk_npy_path, allow_pickle=True))
            multi_spk_emb = np.stack(multi_spk_emb, axis=0)
            res['multi_spk_emb'] = multi_spk_emb
        except Exception as e:
            # traceback.print_exc()
            print(f"| Skip item. item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        # print(res['multi_spk_emb'].shape)
        return res


