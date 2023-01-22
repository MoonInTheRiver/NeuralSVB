import glob
import importlib
import json
import os

import torch
from tqdm import tqdm
import utils
from modules.voice_conversion.svb_ppg import SVBPPG
from tasks.singing.svb_ppg import SVBPPGTask
import torch.nn.functional as F

from tasks.singing.neural_svb_task import FastSingingDataset
from utils import audio
from utils.hparams import hparams
from vocoders.base_vocoder import get_vocoder_cls


class SVCParaTask(SVBPPGTask):
    def run_model(self, model, sample, tech_prefix, return_output=False):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        mels = sample[f'{tech_prefix}mels']  # [B, T_s, 80]
        pitch = sample[f'{tech_prefix}pitch']
        energy = sample[f'{tech_prefix}energy']
        spk_ids = sample['spk_ids'] if hparams['use_spk_id'] else None
        if tech_prefix == 'prof_':
            tech_ids = torch.ones_like(spk_ids)
        else:
            tech_ids = torch.zeros_like(spk_ids)
        output = self.model(mels_content=mels, mels_timbre=mels,
                            pitch=pitch, energy=energy, spk_ids=spk_ids, tech_ids=tech_ids)
        losses = {}
        self.add_mel_loss(output['mel_out'], mels, losses)
        txt_tokens_ = self.model.train_vc_asr(mels, txt_tokens)
        losses['asr'] = F.cross_entropy(txt_tokens_.transpose(1, 2), txt_tokens, ignore_index=0)
        if not return_output:
            return losses
        else:
            return losses, output

    def validation_step(self, sample, batch_idx):
        tech_prefix = ''   # amateur tech
        switched_tech_prefix = 'prof_'
        nsamples = sample['nsamples']
        gtmels = sample[f'{tech_prefix}mels']
        pitch = sample[f'{tech_prefix}pitch']
        energy = sample[f'{tech_prefix}energy']
        spk_ids = sample[f'{tech_prefix}spk_ids']
        f0s = sample[f'{tech_prefix}f0']
        switched_f0s = sample[f'{switched_tech_prefix}f0']
        if switched_tech_prefix == 'prof_':
            switched_tech_ids = torch.ones_like(spk_ids)
        else:
            switched_tech_ids = torch.zeros_like(spk_ids)

        vmin = hparams['mel_vmin']
        vmax = hparams['mel_vmax']
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(self.model, sample, tech_prefix=tech_prefix, return_output=True)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = nsamples
        mel_out = model_out['mel_out']
        outputs = utils.tensors_to_scalars(outputs)
        if self.global_step % hparams['valid_infer_interval'] == 0 \
                and batch_idx < hparams['num_valid_plots']:
            self.plot_mel(batch_idx, gtmels, mel_out)
            switched_mel_out = self.model(mels_content=gtmels,
                                 mels_timbre=gtmels, pitch=pitch,
                                 energy=energy, spk_ids=spk_ids, tech_ids=switched_tech_ids)['mel_out']
            if self.vocoder is None:
                self.vocoder = get_vocoder_cls(hparams)()
            sampling_rate = hparams['audio_sample_rate']

            origin_wav_out = self.vocoder.spec2wav(mel_out[0].cpu(), f0=f0s[0].cpu())
            switched_wav_out = self.vocoder.spec2wav(switched_mel_out[0].cpu(), f0=switched_f0s[0].cpu())
            self.logger.add_audio(f'origin_wavout_{batch_idx}', origin_wav_out, self.global_step, sampling_rate)
            self.logger.add_audio(f'switched_wavout_{batch_idx}', switched_wav_out, self.global_step, sampling_rate)
            self.plot_mel(batch_idx, mel_out, switched_mel_out, name=f'switch_compare_{batch_idx}')
            # self.logger.add_figure(
            #     f'switched_melout_{batch_idx}', spec_to_figure(switched_mel_out[0], vmin, vmax), self.global_step)
            gt_wav = self.vocoder.spec2wav(gtmels[0].cpu(), f0=f0s[0].cpu())
            self.logger.add_audio(f'gt_wav_{batch_idx}', gt_wav, self.global_step, sampling_rate)
        return outputs

