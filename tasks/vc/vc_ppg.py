import glob
import importlib
import json
import os

import torch
from tqdm import tqdm
import utils
from modules.voice_conversion.vc_ppg import VCPPG
from tasks.tts.fs2_adv import FastSpeech2AdvTask
import torch.nn.functional as F

from tasks.tts.fs2_utils import FastSpeechDataset
from utils import audio
from utils.hparams import hparams
from utils.plot import spec_to_figure
from vocoders.base_vocoder import get_vocoder_cls



def load_test_inputs(inp_wav_paths, item_names, spk_id=0):
    sizes = []
    items = []
    binarizer_cls = hparams.get("binarizer_cls", 'data_gen.tts.base_binarizer.BaseBinarizer')
    pkg = ".".join(binarizer_cls.split(".")[:-1])
    cls_name = binarizer_cls.split(".")[-1]
    binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
    binarization_args = {}
    binarization_args.update(hparams['binarization_args'])
    binarization_args['with_txt'] = False

    for wav_fn, item_name in zip(inp_wav_paths, item_names):
        ph = txt = tg_fn = ''
        encoder = None, None
        item = binarizer_cls.process_item(
            item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder, binarization_args)
        items.append(item)
        item['phone'] = [1]
        item['word_tokens'] = [1]
        sizes.append(1)
    return items, sizes


class FastSpeechWordDataset(FastSpeechDataset):
    def __getitem__(self, index):
        sample = super(FastSpeechWordDataset, self).__getitem__(index)
        item = self._get_item(index)
        sample["word_tokens"] = torch.LongTensor(item["word_tokens"])
        return sample

    def collater(self, samples):
        batch = super(FastSpeechWordDataset, self).collater(samples)
        word_tokens = utils.collate_1d([s['word_tokens'] for s in samples], 0)
        batch['word_tokens'] = word_tokens
        return batch


class VCPPGTask(FastSpeech2AdvTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = FastSpeechWordDataset
        self.train_ds = self.dataset_cls('train')

    def build_tts_model(self):
        data_dir = hparams['binary_data_dir']
        word_list_file = os.path.join(data_dir, 'word_set.json')
        word_list = json.load(open(word_list_file))
        self.model = VCPPG(len(word_list) + 10)

    def run_model(self, model, sample, return_output=False):
        txt_tokens = sample['word_tokens']  # [B, T_t]
        mels = sample['mels']  # [B, T_s, 80]
        pitch = sample['pitch']
        energy = sample['energy']
        spk_ids = sample['spk_ids'] if hparams['use_spk_id'] else None
        output = self.model(mels_content=mels, mels_timbre=mels,
                            pitch=pitch, energy=energy, spk_ids=spk_ids)
        losses = {}
        self.add_mel_loss(output['mel_out'], mels, losses)
        txt_tokens_ = self.model.train_vc_asr(mels, txt_tokens)
        losses['asr'] = F.cross_entropy(txt_tokens_.transpose(1, 2), txt_tokens, ignore_index=0)
        if not return_output:
            return losses
        else:
            return losses, output

    def validation_step(self, sample, batch_idx):
        vmin = hparams['mel_vmin']
        vmax = hparams['mel_vmax']
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        mel_out = model_out['mel_out']
        outputs = utils.tensors_to_scalars(outputs)
        if self.global_step % hparams['valid_infer_interval'] == 0 \
                and batch_idx < hparams['num_valid_plots']:
            self.plot_mel(batch_idx, sample['mels'], mel_out)
            timbre_sample = self.train_ds.collater(
                [self.train_ds[hparams['valid_mel_timbre_id']]])
            mel_timbre = timbre_sample['mels'].to(mel_out.device)
            spk_ids = timbre_sample['spk_ids'].cuda() if hparams['use_spk_id'] else None
            mel_out = self.model(mels_content=sample['mels'],
                                 mels_timbre=mel_timbre, pitch=sample['pitch'],
                                 energy=sample['energy'], spk_ids=spk_ids)['mel_out']
            if self.vocoder is None:
                self.vocoder = get_vocoder_cls(hparams)()
            wav_pred = self.vocoder.spec2wav(mel_out[0].cpu())
            sampling_rate = hparams['audio_sample_rate']
            self.logger.add_audio(f'wav_{batch_idx}', wav_pred, self.global_step, sampling_rate)
            self.logger.add_figure(
                f'mel_infer_{batch_idx}', spec_to_figure(mel_out[0], vmin, vmax), self.global_step)
            if self.global_step == 0:
                wav_gt = self.vocoder.spec2wav(sample['mels'][0].cpu())
                self.logger.add_audio(f'wav_gt_{batch_idx}', wav_gt, self.global_step, sampling_rate)
                wav_timbre = self.vocoder.spec2wav(mel_timbre[0].cpu())
                self.logger.add_audio(f'wav_timbre_{batch_idx}', wav_timbre, self.global_step, sampling_rate)
        return outputs

    def test_start(self):
        vocoder = get_vocoder_cls(hparams)()
        test_input_dir = hparams['test_input_dir']
        assert test_input_dir != ''
        content_wav_fns = glob.glob(f'{test_input_dir}/content_inputs/*.wav')
        content_item_names = [os.path.basename(f)[:-4] for f in content_wav_fns]
        content_ds, content_ds_size = load_test_inputs(
            content_wav_fns, content_item_names)
        timbre_wav_fns = glob.glob(f'{test_input_dir}/timbre_inputs/*.wav')
        timbre_item_names = [os.path.basename(f)[:-4] for f in timbre_wav_fns]
        timbre_ds, timbre_ds_size = load_test_inputs(timbre_wav_fns, timbre_item_names)
        content_ds = self.dataset_cls('test', False, content_ds, content_ds_size)
        timbre_ds = self.dataset_cls('test', False, timbre_ds, timbre_ds_size)
        gen_dir = os.path.join(hparams['work_dir'],
                               f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        os.makedirs(gen_dir, exist_ok=True)
        for i in tqdm(range(len(content_ds))):
            sample_content = content_ds.collater([content_ds[i]])
            mels_content = sample_content['mels'].cuda()
            pitch = sample_content['pitch'].cuda()
            energy = sample_content['energy'].cuda()
            item_name_c = sample_content['item_name'][0]
            for j in tqdm(range(len(timbre_ds))):
                sample_timbre = timbre_ds.collater([timbre_ds[j]])
                mel_timbre = sample_timbre['mels'].cuda()
                item_name_t = sample_timbre['item_name'][0]
                spk_ids = sample_timbre['spk_ids'].cuda() if hparams['use_spk_id'] else None
                mel_out = self.model(
                    mels_content=mels_content,
                    mels_timbre=mel_timbre, pitch=pitch, energy=energy,
                    spk_ids=spk_ids)['mel_out']
                wav_out = vocoder.spec2wav(mel_out[0].cpu())
                audio.save_wav(wav_out,
                               f'{gen_dir}/C[{item_name_c}]_T[{item_name_t}].wav',
                               hparams['audio_sample_rate'])
        return 'EXIT'
