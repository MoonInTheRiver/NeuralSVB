import glob
import json
import os

import torch
from tqdm import tqdm
import utils
from modules.voice_conversion.svb_ppg import SVBPPG
from utils.pitch_utils import denorm_f0

from tasks.tts.fs2_adv import FastSpeech2AdvTask
from tasks.vc.vc_ppg import load_test_inputs
import torch.nn.functional as F

from tasks.singing.neural_svb_task import FastSingingDataset
from utils import audio
from utils.hparams import hparams
from vocoders.base_vocoder import get_vocoder_cls
import numpy as np


class SVBPPGTask(FastSpeech2AdvTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = FastSingingDataset
        self.train_ds = self.dataset_cls('train')

    def build_tts_model(self):
        data_dir = hparams['binary_data_dir']
        phone_list_file = os.path.join(data_dir, 'phone_set.json')
        phone_list = json.load(open(phone_list_file))
        self.model = SVBPPG(len(phone_list) + 10)

    def _training_step(self, sample, batch_idx, optimizer_idx):

        log_outputs = {}
        loss_weights = {}
        disc_start = hparams['mel_gan'] and self.global_step > hparams["disc_start_steps"] and \
                     hparams['lambda_mel_adv'] > 0
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            tech_prefix = ['', 'prof_'][np.random.randint(low=0, high=2)]  # random choo
            log_outputs, model_out = self.run_model(self.model, sample, tech_prefix=tech_prefix, return_output=True)
            self.model_out_gt = self.model_out = \
                {k: v.detach() for k, v in model_out.items() if isinstance(v, torch.Tensor)}
            self.model_out_gt['tech_prefix'] = tech_prefix
            if disc_start:
                self.disc_cond_gt = self.model_out['decoder_inp'].detach() if hparams['use_cond_disc'] else None
                self.disc_cond = disc_cond = self.model_out['decoder_inp'].detach() \
                    if hparams['use_cond_disc'] else None
                mel_p = model_out['mel_out']
                if hasattr(self.model, 'out2mel'):
                    mel_p = self.model.out2mel(mel_p)
                o_ = self.mel_disc(mel_p, disc_cond)
                p_, pc_ = o_['y'], o_['y_c']
                if p_ is not None:
                    log_outputs['a'] = self.mse_loss_fn(p_, p_.new_ones(p_.size()))
                    loss_weights['a'] = hparams['lambda_mel_adv']
                if pc_ is not None:
                    log_outputs['ac'] = self.mse_loss_fn(pc_, pc_.new_ones(pc_.size()))
                    loss_weights['ac'] = hparams['lambda_mel_adv']
        else:
            #######################
            #    Discriminator    #
            #######################
            if disc_start and self.global_step % hparams['disc_interval'] == 0:
                assert not hparams['rerun_gen'], 'rerun_gen is not built.'
                model_out = self.model_out_gt
                tech_prefix = model_out['tech_prefix']
                mel_g = sample[f'{tech_prefix}mels']
                mel_p = model_out['mel_out']
                if hasattr(self.model, 'out2mel'):
                    mel_p = self.model.out2mel(mel_p)
                o = self.mel_disc(mel_g, self.disc_cond_gt)
                p, pc = o['y'], o['y_c']
                o_ = self.mel_disc(mel_p, self.disc_cond)
                p_, pc_ = o_['y'], o_['y_c']
                if p_ is not None:
                    log_outputs["r"] = self.mse_loss_fn(p, p.new_ones(p.size()))
                    log_outputs["f"] = self.mse_loss_fn(p_, p_.new_zeros(p_.size()))
                if pc_ is not None:
                    log_outputs["rc"] = self.mse_loss_fn(pc, pc.new_ones(pc.size()))
                    log_outputs["fc"] = self.mse_loss_fn(pc_, pc_.new_zeros(pc_.size()))
            if len(log_outputs) == 0:
                return None
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in log_outputs.items()])
        log_outputs['bs'] = sample['mels'].shape[0]
        return total_loss, log_outputs

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
        energy = sample[f'{tech_prefix}energy']
        spk_ids = sample[f'{tech_prefix}spk_ids']
        # pitch = sample[f'{tech_prefix}pitch']
        f0s = denorm_f0(sample[f'{tech_prefix}f0'], sample[f'{tech_prefix}uv'], hparams)

        switched_pitch = sample[f'{switched_tech_prefix}pitch']
        switched_f0s = denorm_f0(sample[f'{switched_tech_prefix}f0'],sample[f'{switched_tech_prefix}uv'], hparams)

        if switched_tech_prefix == 'prof_':
            switched_tech_ids = torch.ones_like(spk_ids)
        else:
            switched_tech_ids = torch.zeros_like(spk_ids)

        # vmin = hparams['mel_vmin']
        # vmax = hparams['mel_vmax']
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
                                 mels_timbre=gtmels, pitch=switched_pitch,
                                 energy=energy, spk_ids=spk_ids, tech_ids=switched_tech_ids)['mel_out']
            if self.vocoder is None:
                self.vocoder = get_vocoder_cls(hparams)()
            sampling_rate = hparams['audio_sample_rate']

            origin_wav_out = self.vocoder.spec2wav(mel_out[0].cpu(), f0=f0s[0].cpu())
            switched_wav_out = self.vocoder.spec2wav(switched_mel_out[0].cpu(), f0=switched_f0s[0].cpu())
            self.logger.add_audio(f'origin_wavout_{batch_idx}', origin_wav_out, self.global_step, sampling_rate)
            self.logger.add_audio(f'switched_wavout_{batch_idx}', switched_wav_out, self.global_step, sampling_rate)
            pad_num = max(switched_mel_out.shape[1] - mel_out.shape[1], 0)  # if switched is longger, pad origin
            padded_mel_out = F.pad(mel_out, [0, 0, 0, pad_num])[:, :switched_mel_out.shape[1], :]  # if origin is longer, cut origin
            self.plot_mel(batch_idx, padded_mel_out, switched_mel_out, name=f'switch_compare_{batch_idx}')
            # self.logger.add_figure(
            #     f'switched_melout_{batch_idx}', spec_to_figure(switched_mel_out[0], vmin, vmax), self.global_step)
            gt_wav = self.vocoder.spec2wav(gtmels[0].cpu(), f0=f0s[0].cpu())
            self.logger.add_audio(f'gt_wav_{batch_idx}', gt_wav, self.global_step, sampling_rate)
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