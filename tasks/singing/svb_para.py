import glob
import importlib
import json
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from multiprocessing.pool import Pool
from tasks.singing.svb_base import SVBPPGTask, FastSpeech2AdvTask
from tasks.singing.neural_svb_task import FastSingingDataset
from modules.voice_conversion.svb_ppg import ParaSVBPPG, ParaPPGConstraint, ParaPPGPreExp, ParaAlignedPPG
from modules.fastspeech.multi_window_disc import Discriminator
from vocoders.base_vocoder import get_vocoder_cls
from data_gen.tts.data_gen_utils import get_pitch

import utils
from utils import audio
from utils.hparams import hparams
from utils.plot import spec_to_figure
from utils.pitch_utils import norm_interp_f0, denorm_f0, f0_to_coarse


class FastSingingF0AlignDataset(FastSingingDataset):
    def __getitem__(self, index):
        sample = super(FastSingingF0AlignDataset, self).__getitem__(index)
        item = self._get_item(index)
        a2p_f0_alignment = torch.LongTensor(item['a2p_f0_alignment'])
        p2a_f0_alignment = torch.LongTensor(item['p2a_f0_alignment'])

        a2p_f0_alignment = a2p_f0_alignment[:sample['prof_pitch'].shape[0]].clamp(max=sample['pitch'].shape[0] - 1)
        p2a_f0_alignment = p2a_f0_alignment[:sample['pitch'].shape[0]].clamp(max=sample['prof_pitch'].shape[0] - 1)
        # if not a2p_f0_alignment.shape == sample['prof_pitch'].shape:
        #     print('a2p F0 alignment with unmatched shape: ', a2p_f0_alignment.shape, sample['prof_pitch'].shape, sample['prof_mel'].shape)
        # if not p2a_f0_alignment.shape == sample['pitch'].shape:
        #     print('p2a F0 alignment with unmatched shape: ', p2a_f0_alignment.shape, sample['pitch'].shape, sample['mel'].shape)
        assert a2p_f0_alignment.shape == sample['prof_pitch'].shape, (
        'a2p F0 alignment with unmatched shape: ', a2p_f0_alignment.shape, sample['prof_pitch'].shape,
        sample['prof_mel'].shape)
        assert p2a_f0_alignment.shape == sample['pitch'].shape, (
        'p2a F0 alignment with unmatched shape: ', p2a_f0_alignment.shape, sample['pitch'].shape, sample['mel'].shape)
        sample['a2p_f0_alignment'] = a2p_f0_alignment
        sample['p2a_f0_alignment'] = p2a_f0_alignment
        if 'multi_spk_emb' in item.keys():
            sample['multi_spk_emb'] = torch.LongTensor(item['multi_spk_emb'])
        return sample

    def collater(self, samples):
        batch = super(FastSingingF0AlignDataset, self).collater(samples)
        batch['a2p_f0_alignment'] = utils.collate_1d([s['a2p_f0_alignment'] for s in samples])
        batch['p2a_f0_alignment'] = utils.collate_1d([s['p2a_f0_alignment'] for s in samples])
        if 'multi_spk_emb' in samples[0].keys():
            batch['multi_spk_emb'] = utils.collate_2d([s['multi_spk_emb'] for s in samples])
        return batch


class SVCParaTask(FastSpeech2AdvTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = FastSingingF0AlignDataset
        self.concurrent_ways = hparams['concurrent_ways'].split(',')
        self.train_ds = self.dataset_cls('train')

    def build_tts_model(self):
        data_dir = hparams['binary_data_dir']
        phone_list_file = os.path.join(data_dir, 'phone_set.json')
        phone_list = json.load(open(phone_list_file))
        self.model = ParaSVBPPG(len(phone_list) + 10)

    def run_model(self, model, sample, tech_prefix=None, return_output=False, infer=False):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        # spk_ids = sample['spk_ids'] if hparams['use_spk_id'] else None

        amateur_mels = sample['mels']  # [B, T_s, 80]
        amateur_pitch = sample['pitch']
        # amateur_energy = sample['energy']
        amateur_tech_id = torch.zeros([amateur_mels.shape[0]], dtype=torch.long, device=amateur_mels.device)

        prof_mels = sample['prof_mels']
        prof_pitch = sample['prof_pitch']
        # prof_energy = sample['prof_energy']
        prof_tech_id = torch.ones([prof_mels.shape[0]], dtype=torch.long, device=prof_mels.device)

        a2p_alignment = sample['a2p_f0_alignment']
        p2a_alignment = sample['p2a_f0_alignment']

        multi_spk_emb = sample['multi_spk_emb']

        output = {}
        #  暂时不给energy.
        if 'a2a' in self.concurrent_ways:
            a2a_output = self.model(mels_content=amateur_mels, mels_timbre=amateur_mels, pitch=amateur_pitch,
                                    energy=None, spk_ids=multi_spk_emb, tech_ids=amateur_tech_id, infer=infer)
            output['a2a'] = a2a_output
        if 'p2p' in self.concurrent_ways:
            p2p_output = self.model(mels_content=prof_mels, mels_timbre=prof_mels, pitch=prof_pitch,
                                    energy=None, spk_ids=multi_spk_emb, tech_ids=prof_tech_id, infer=infer)
            output['p2p'] = p2p_output

        if 'a2p' in self.concurrent_ways:
            a2p_output = self.model(mels_content=amateur_mels, mels_timbre=amateur_mels, pitch=prof_pitch,
                                    energy=None, spk_ids=multi_spk_emb, tech_ids=prof_tech_id,
                                    conversion_alignment=a2p_alignment, infer=infer)
            output['a2p'] = a2p_output

        if 'p2a' in self.concurrent_ways:
            p2a_output = self.model(mels_content=prof_mels, mels_timbre=prof_mels, pitch=amateur_pitch,
                                    energy=None, spk_ids=multi_spk_emb, tech_ids=amateur_tech_id,
                                    conversion_alignment=p2a_alignment, infer=infer)
            output['p2a'] = p2a_output

        losses = {}
        for way in self.concurrent_ways:
            mel_g = self.get_corresponding_gtmel(way, sample)
            self.add_mel_loss(output[way]['mel_out'], mel_g, losses, postfix=way)

        self.add_asr_losses(amateur_mels, prof_mels, txt_tokens, losses, a2p_alignment, p2a_alignment)
        if not return_output:
            return losses
        else:
            return losses, output

    def gen_cheat_disc(self, way, model_out_with_gradient, log_outputs, loss_weights):
        self.disc_cond_gt[way] = self.model_out[way]['decoder_inp'].detach() if hparams['use_cond_disc'] else None
        self.disc_cond[way] = disc_cond_current_way = self.model_out[way]['decoder_inp'].detach() \
            if hparams['use_cond_disc'] else None
        mel_p = model_out_with_gradient[way]['mel_out']
        if hasattr(self.model, 'out2mel'):
            mel_p = self.model.out2mel(mel_p)
        o_ = self.mel_disc(mel_p, disc_cond_current_way)
        p_, pc_ = o_['y'], o_['y_c']
        if p_ is not None:
            log_outputs[f'{way}_a'] = self.mse_loss_fn(p_, p_.new_ones(p_.size()))
            loss_weights[f'{way}_a'] = hparams['lambda_mel_adv']
        if pc_ is not None:
            log_outputs[f'{way}_ac'] = self.mse_loss_fn(pc_, pc_.new_ones(pc_.size()))
            loss_weights[f'{way}_ac'] = hparams['lambda_mel_adv']

    def get_corresponding_gtmel(self, way, sample, return_tgtprefix=False):
        tgt_prefix = ''
        if way == 'a2a':
            mel_g = sample['mels']
        elif way == 'p2p':
            mel_g = sample['prof_mels']
            tgt_prefix = 'prof_'
        elif way == 'a2p':
            mel_g = sample['prof_mels']
            tgt_prefix = 'prof_'
        elif way == 'p2a':
            mel_g = sample['mels']
        else:
            mel_g = None
        if return_tgtprefix:
            return mel_g, tgt_prefix
        return mel_g

    def disc_judge_gen(self, way, sample, log_outputs):
        if hparams['rerun_gen']:
            exit(1)  # not allowed
        else:
            model_out = self.model_out_gt  # 这里是没有gradient的model_out.
        mel_g = self.get_corresponding_gtmel(way, sample)
        mel_p = model_out[way]['mel_out']
        if hasattr(self.model, 'out2mel'):
            mel_p = self.model.out2mel(mel_p)
        o = self.mel_disc(mel_g, self.disc_cond_gt[way])
        p, pc = o['y'], o['y_c']
        o_ = self.mel_disc(mel_p, self.disc_cond[way])
        p_, pc_ = o_['y'], o_['y_c']
        if p_ is not None:
            log_outputs[f"{way}_r"] = self.mse_loss_fn(p, p.new_ones(p.size()))
            log_outputs[f"{way}_f"] = self.mse_loss_fn(p_, p_.new_zeros(p_.size()))
        if pc_ is not None:
            log_outputs[f"{way}_rc"] = self.mse_loss_fn(pc, pc.new_ones(pc.size()))
            log_outputs[f"{way}_fc"] = self.mse_loss_fn(pc_, pc_.new_zeros(pc_.size()))

    def _training_step(self, sample, batch_idx, optimizer_idx):
        log_outputs = {}
        loss_weights = {}
        disc_start = hparams['mel_gan'] and self.global_step > hparams["disc_start_steps"] and \
                     hparams['lambda_mel_adv'] > 0
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            log_outputs, model_out = self.run_model(self.model, sample, return_output=True)
            self.model_out = {}  # dict of dict
            for way in model_out.keys():
                self.model_out[way] = {}
                for k, v in model_out[way].items():
                    if isinstance(v, torch.Tensor):
                        self.model_out[way][k] = v.detach()
            self.model_out_gt = self.model_out

            if disc_start:
                self.disc_cond_gt = {}
                self.disc_cond = {}
                for way in self.concurrent_ways:
                    self.gen_cheat_disc(way, model_out, log_outputs, loss_weights)
        else:
            #######################
            #    Discriminator    #
            #######################
            if disc_start and self.global_step % hparams['disc_interval'] == 0:
                for way in self.concurrent_ways:
                    self.disc_judge_gen(way, sample, log_outputs)

            if len(log_outputs) == 0:
                return None
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in log_outputs.items()])
        log_outputs['bs'] = sample['mels'].shape[0]
        return total_loss, log_outputs

    def pitch_debugging(self, mel_pred, mel_gt, f0_gt, uv_gt, name):
        fig = plt.figure(figsize=(12, 6))
        spec_vmin = hparams['mel_vmin']
        spec_vmax = hparams['mel_vmax']

        f0_gt = denorm_f0(f0_gt, uv_gt, hparams)
        f0_gt = f0_gt[0].cpu().numpy()

        mel = (torch.cat([mel_gt, mel_pred], -1))[0].cpu().numpy()
        f0_gt = f0_gt / 10 * (f0_gt > 0)
        f0_another = f0_gt + 80  # to be changed

        plt.pcolor(mel.T, vmin=spec_vmin, vmax=spec_vmax)
        plt.plot(f0_gt, c='white', linewidth=1, alpha=0.6)
        plt.plot(f0_another, c='red', linewidth=1, alpha=0.6)
        self.logger.add_figure(name, fig, self.global_step)

    def validation_step(self, sample, batch_idx):
        prof_mel = sample['prof_mels']
        prof_f0s = denorm_f0(sample['prof_f0'], sample['prof_uv'], hparams)

        amateur_mel = sample['mels']
        amateur_f0s = denorm_f0(sample['f0'], sample['uv'], hparams)

        f0s = {
            'a2a': amateur_f0s,
            'p2p': prof_f0s,
            'a2p': prof_f0s,
            'p2a': amateur_f0s,
        }
        vmin = hparams['mel_vmin']
        vmax = hparams['mel_vmax']
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']

        outputs = utils.tensors_to_scalars(outputs)
        if self.global_step % hparams['valid_infer_interval'] == 0 \
                and batch_idx < hparams['num_valid_plots']:
            if self.vocoder is None:
                self.vocoder = get_vocoder_cls(hparams)()
            sampling_rate = hparams['audio_sample_rate']

            for way in self.concurrent_ways:
                _wav_out = self.vocoder.spec2wav(model_out[way]['mel_out'][0].cpu(),
                                                 f0=f0s[way][0].cpu())  # , f0=amateur_f0s[0].cpu())
                self.logger.add_audio(f'{way}_wavout_{batch_idx}', _wav_out, self.global_step, sampling_rate)
                mel_g, tgtprefix = self.get_corresponding_gtmel(way, sample, return_tgtprefix=True)
                self.plot_mel(batch_idx, mel_g, model_out[way]['mel_out'], name=f'{way}_gt_{batch_idx}')

                ### here for debugging
                self.pitch_debugging(model_out[way]['mel_out'], mel_g, sample[f'{tgtprefix}f0'],
                                     sample[f'{tgtprefix}uv'], name=f'{way}_f0_w_mel_{batch_idx}')

            gt_a_wav_out = self.vocoder.spec2wav(amateur_mel[0].cpu(),
                                                 f0=amateur_f0s[0].cpu())  # , f0=amateur_f0s[0].cpu())
            gt_p_wav_out = self.vocoder.spec2wav(prof_mel[0].cpu(), f0=prof_f0s[0].cpu())  # , f0=prof_f0s[0].cpu())
            self.logger.add_audio(f'gt_a_wav_{batch_idx}', gt_a_wav_out, self.global_step, sampling_rate)
            self.logger.add_audio(f'gt_p_wav_{batch_idx}', gt_p_wav_out, self.global_step, sampling_rate)

        return outputs

    ### inference logic
    def test_step(self, sample, batch_idx):
        _, model_out = self.run_model(self.model, sample, return_output=True)

        prof_f0s = denorm_f0(sample['prof_f0'], sample['prof_uv'], hparams)
        amateur_f0s = denorm_f0(sample['f0'], sample['uv'], hparams)
        f0s = {
            'a2a_f0s': amateur_f0s,
            'p2p_f0s': prof_f0s,
            'a2p_f0s': prof_f0s,
            'p2a_f0s': amateur_f0s
        }
        mels = {}
        for way in self.concurrent_ways:
            mels[f'{way}_mels'] = model_out[way]['mel_out']

        sample.update(f0s)
        sample.update(mels)
        sample['outputs'] = sample['a2p_mels']
        return self.after_infer(sample)

    def after_infer(self, predictions, sil_start_frame=0):
        self.results_id = 0
        predictions = utils.unpack_dict_to_list(predictions)
        assert len(predictions) == 1, 'Only support batch_size=1 in inference.'
        prediction = predictions[0]
        prediction = utils.tensors_to_np(prediction)
        item_name = prediction.get('item_name')
        text = prediction.get('text')

        str_phs = None
        if self.phone_encoder is not None and 'txt_tokens' in prediction:
            str_phs = self.phone_encoder.decode(prediction['txt_tokens'], strip_padding=True)

        gen_dir = os.path.join(hparams['work_dir'],
                               f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        base_fn = f'[{self.results_id:06d}][{item_name}][%s]'
        if text is not None:
            base_fn += text.replace(":", "%3A")[:80]
        base_fn = base_fn.replace(' ', '_')

        wavs_dict = {
            'gt_a_wavout': self.vocoder.spec2wav(prediction['mels'], f0=prediction['a2a_f0s']),
            'gt_p_wavout': self.vocoder.spec2wav(prediction['prof_mels'], f0=prediction['p2p_f0s']),
        }

        for way in self.concurrent_ways:
            wavs_dict[f'{way}_wavout'] = self.vocoder.spec2wav(prediction[f'{way}_mels'], f0=prediction[f'{way}_f0s'])

        os.makedirs(gen_dir, exist_ok=True)
        os.makedirs(f'{gen_dir}/mels', exist_ok=True)
        os.makedirs(f'{gen_dir}/wavs', exist_ok=True)
        for key in wavs_dict:
            os.makedirs(f'{gen_dir}/wavs/{key}', exist_ok=True)

        self.saving_results_futures.append(
            self.saving_result_pool.apply_async(self.save_result, args=[
                wavs_dict, base_fn % 'P', gen_dir, str_phs]))

        self.results_id += 1
        return {
            'item_name': item_name,
            'text': text,
        }

    @staticmethod
    def save_result(wavs_dict, base_fn, gen_dir, str_phs=None, mels_dict=None):
        for key in wavs_dict:
            if hparams.get('disable_map') is not None and hparams['disable_map']:
                audio.save_wav(wavs_dict[key], f'{gen_dir}/wavs/disable_map_{key}/{base_fn}.wav',
                               hparams['audio_sample_rate'],
                               norm=hparams['out_wav_norm'])
            else:
                audio.save_wav(wavs_dict[key], f'{gen_dir}/wavs/{key}/{base_fn}.wav', hparams['audio_sample_rate'],
                               norm=hparams['out_wav_norm'])
        if mels_dict is not None:
            for key in mels_dict:
                if hparams.get('disable_map') is not None and hparams['disable_map']:
                    np.save(f'{gen_dir}/mels/disable_map_{key}/{base_fn}.npy', mels_dict[key])
                else:
                    np.save(f'{gen_dir}/mels/{key}/{base_fn}.npy', mels_dict[key])

    ### losses
    def add_asr_losses(self, amateur_mels, prof_mels, txt_tokens, losses, a2p_alignment, p2a_alignment):
        input_content_a, input_content_p = False, False
        for way in self.concurrent_ways:
            if way.startswith('a'):
                input_content_a = True
            elif way.startswith('p'):
                input_content_p = True
        if input_content_a:
            txt_tokens_a = self.model.train_vc_asr(amateur_mels, txt_tokens)
            losses['asr_a'] = F.cross_entropy(txt_tokens_a.transpose(1, 2), txt_tokens, ignore_index=0)
        if input_content_p:
            txt_tokens_p = self.model.train_vc_asr(prof_mels, txt_tokens)
            losses['asr_p'] = F.cross_entropy(txt_tokens_p.transpose(1, 2), txt_tokens, ignore_index=0)


class ParaPPGConstraintTask(SVCParaTask):
    def build_tts_model(self):
        data_dir = hparams['binary_data_dir']
        phone_list_file = os.path.join(data_dir, 'phone_set.json')
        phone_list = json.load(open(phone_list_file))
        self.model = ParaPPGConstraint(len(phone_list) + 10)

    def ppg_loss(self, src_ppg, tgt_ppg, tgt_mask):
        # src_ppg : B x T x H
        # tgt_ppg : B x T x H
        # tgt_mel : B x T x n_mels
        assert src_ppg.shape == tgt_ppg.shape, (src_ppg.shape, tgt_ppg.shape)
        mse_loss = F.mse_loss(src_ppg, tgt_ppg, reduction='none')
        mse_loss = (mse_loss * tgt_mask).sum() / tgt_mask.sum()
        return mse_loss

    def add_asr_losses(self, amateur_mels, prof_mels, txt_tokens, losses, a2p_alignment, p2a_alignment):
        txt_tokens_a, h_content_a = self.model.train_vc_asr(amateur_mels, txt_tokens,
                                                            a2p_alignment)  # expand h_content_a to shape-p
        txt_tokens_p, h_content_p = self.model.train_vc_asr(prof_mels, txt_tokens)
        # asr losses
        losses['asr_a'] = F.cross_entropy(txt_tokens_a.transpose(1, 2), txt_tokens, ignore_index=0)
        losses['asr_p'] = F.cross_entropy(txt_tokens_p.transpose(1, 2), txt_tokens, ignore_index=0)
        # ppg constraint loss
        a2p_T_div_scale = h_content_p.shape[1]  # [B, T // scale]
        scale = np.prod(hparams['mel_strides'])
        mel_lengths = (prof_mels.abs().sum(-1).ne(0).float().sum(-1) / scale).long()  # [B, ]
        tgt_mask = (torch.arange(a2p_T_div_scale)[None, :].to(h_content_p.device) < mel_lengths[:,
                                                                                    None]).float()  # [B, T // scale]
        h_content_a = h_content_a[:, :a2p_T_div_scale]  # [B, , T // scale]
        # tgt_mel.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, tgt_ppg.shape[-1])
        # print(tgt_mask[-10:])
        # print(tgt_mask.shape, mel_lengths, h_content_p.shape, h_content_a.shape, amateur_mels.shape, prof_mels.shape)
        losses['ppg_constraint'] = self.ppg_loss(h_content_a, h_content_p.detach(),
                                                 tgt_mask=tgt_mask[:, :, None].repeat(1, 1,
                                                                                      hparams['hidden_size'])) * 0.1


class ParaPPGPreExpTask(SVCParaTask):
    def build_tts_model(self):
        data_dir = hparams['binary_data_dir']
        phone_list_file = os.path.join(data_dir, 'phone_set.json')
        phone_list = json.load(open(phone_list_file))
        self.model = ParaPPGPreExp(len(phone_list) + 10)

    ### losses
    def add_asr_losses(self, amateur_mels, prof_mels, txt_tokens, losses, a2p_alignment, p2a_alignment):
        txt_tokens_a = self.model.train_vc_asr(amateur_mels, txt_tokens, a2p_alignment)  # expand mel_a to shape-p
        # asr losses
        losses['asr_a'] = F.cross_entropy(txt_tokens_a.transpose(1, 2), txt_tokens, ignore_index=0)


class ParaAlignedPPGTask(SVCParaTask):
    def build_tts_model(self):
        data_dir = hparams['binary_data_dir']
        phone_list_file = os.path.join(data_dir, 'phone_set.json')
        phone_list = json.load(open(phone_list_file))
        self.model = ParaAlignedPPG(len(phone_list) + 10)


class ParaPPGPretrainedTask(SVCParaTask):
    def build_model(self):
        self.build_tts_model()
        utils.load_ckpt(self.model.vc_asr, hparams['pretrain_asr_ckpt'], model_name='model')
        self.model.vc_asr.eval()
        for param in self.model.vc_asr.parameters():
            param.requires_grad = False
        self.watch_asr_loss = False

        if hparams['load_ckpt'] != '':
            self.load_ckpt(hparams['load_ckpt'], strict=False)
        utils.print_arch(self.model, 'Generator')
        self.build_disc_model()

        if not hasattr(self, 'gen_params'):
            self.gen_params = [p for name, p in self.model.named_parameters() if
                               'vc_asr' not in name]  # list(self.model.parameters())
            # print([(name, p.requires_grad) for name, p in self.model.named_parameters() if
            #                    'vc_asr' in name])
        return self.model

    def configure_optimizers(self):
        if not hasattr(self, 'gen_params'):
            self.gen_params = [p for name, p in self.model.named_parameters() if
                               'vc_asr' not in name]  # list(self.model.parameters())
        optimizer_gen = torch.optim.AdamW(
            self.gen_params,
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])
        optimizer_disc = torch.optim.AdamW(
            self.disc_params,
            lr=hparams['disc_lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            **hparams["discriminator_optimizer_params"]) if len(self.disc_params) > 0 else None
        self.scheduler = self.build_scheduler({'gen': optimizer_gen, 'disc': optimizer_disc})
        return [optimizer_gen, optimizer_disc]

    def validation_step(self, sample, batch_idx):
        prof_mel = sample['prof_mels']
        prof_f0s = denorm_f0(sample['prof_f0'], sample['prof_uv'], hparams)

        amateur_mel = sample['mels']
        amateur_f0s = denorm_f0(sample['f0'], sample['uv'], hparams)

        f0s = {
            'a2a': amateur_f0s,
            'p2p': prof_f0s,
            'a2p': prof_f0s,
            'p2a': amateur_f0s,
        }
        vmin = hparams['mel_vmin']
        vmax = hparams['mel_vmax']
        outputs = {}
        outputs['losses'] = {}
        self.watch_asr_loss = True
        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True, infer=True)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']

        outputs = utils.tensors_to_scalars(outputs)
        if self.global_step % hparams['valid_infer_interval'] == 0 \
                and batch_idx > 394 and batch_idx < 394 + hparams['num_valid_plots']:
            if self.vocoder is None:
                self.vocoder = get_vocoder_cls(hparams)()
            sampling_rate = hparams['audio_sample_rate']

            for way in self.concurrent_ways:
                _wav_out = self.vocoder.spec2wav(model_out[way]['mel_out'][0].cpu(),
                                                 f0=f0s[way][0].cpu())  # , f0=amateur_f0s[0].cpu())
                self.logger.add_audio(f'{way}_wavout_{batch_idx}', _wav_out, self.global_step, sampling_rate)
                mel_g, tgtprefix = self.get_corresponding_gtmel(way, sample, return_tgtprefix=True)
                self.plot_mel(batch_idx, mel_g, model_out[way]['mel_out'], name=f'{way}_gt_{batch_idx}')

                ### here for debugging
                self.pitch_debugging(model_out[way]['mel_out'], mel_g, sample[f'{tgtprefix}f0'],
                                     sample[f'{tgtprefix}uv'], name=f'{way}_f0_w_mel_{batch_idx}')

            gt_a_wav_out = self.vocoder.spec2wav(amateur_mel[0].cpu(),
                                                 f0=amateur_f0s[0].cpu())  # , f0=amateur_f0s[0].cpu())
            gt_p_wav_out = self.vocoder.spec2wav(prof_mel[0].cpu(), f0=prof_f0s[0].cpu())  # , f0=prof_f0s[0].cpu())
            self.logger.add_audio(f'gt_a_wav_{batch_idx}', gt_a_wav_out, self.global_step, sampling_rate)
            self.logger.add_audio(f'gt_p_wav_{batch_idx}', gt_p_wav_out, self.global_step, sampling_rate)

        return outputs

    ### losses
    def add_asr_losses(self, amateur_mels, prof_mels, txt_tokens, losses, a2p_alignment, p2a_alignment):
        if self.watch_asr_loss:
            txt_tokens_a = self.model.train_vc_asr(amateur_mels, txt_tokens)
            losses['asr_a'] = F.cross_entropy(txt_tokens_a.transpose(1, 2), txt_tokens, ignore_index=0).detach()
            txt_tokens_p = self.model.train_vc_asr(prof_mels, txt_tokens)
            losses['asr_p'] = F.cross_entropy(txt_tokens_p.transpose(1, 2), txt_tokens, ignore_index=0).detach()
            self.watch_asr_loss = False

    # def test_step(self, sample, batch_idx):
    #     return self.validation_step(sample, batch_idx)

    # def test_end(self, outputs):
    #     return self.validation_end(outputs)


class ParaPPGSpkConsistentTask(ParaPPGPretrainedTask):

    def build_disc_model(self):
        super().build_disc_model()
        if hparams['mel_gan']:
            disc_win_num = hparams['disc_win_num']
            h = hparams['mel_disc_hidden_size']
            self.spk_disc = Discriminator(
                time_lengths=[32, 64, 128][:disc_win_num],
                freq_length=80, hidden_size=h, kernel=(3, 3),
                cond_size=hparams['hidden_size'] if self.use_cond_disc else 0,
                norm_type=hparams['disc_norm'], reduction=hparams['disc_reduction']
            )
            self.disc_params += list(self.spk_disc.parameters())
            # utils.print_arch(self.mel_disc, model_name='Mel Disc')

    def build_model(self):
        self.build_tts_model()
        utils.load_ckpt(self.model.vc_asr, hparams['pretrain_asr_ckpt'], model_name='model')
        self.model.vc_asr.eval()
        for param in self.model.vc_asr.parameters():
            param.requires_grad = False
        self.watch_asr_loss = False

        if hparams['load_ckpt'] != '':
            self.load_ckpt(hparams['load_ckpt'], strict=False)
        utils.print_arch(self.model, 'Generator')
        self.build_disc_model()

        if not hasattr(self, 'gen_params'):
            self.gen_params = [p for name, p in self.model.named_parameters() if
                               'vc_asr' not in name]  # list(self.model.parameters())
            # print([(name, p.requires_grad) for name, p in self.model.named_parameters() if
            #                    'vc_asr' in name])
        return self.model

    def gen_cheat_disc(self, way, model_out_with_gradient, log_outputs, loss_weights):
        self.disc_cond_gt[way] = self.model_out[way]['decoder_inp'].detach() if hparams['use_cond_disc'] else None
        self.disc_cond[way] = disc_cond_current_way = self.model_out[way]['decoder_inp'].detach() \
            if hparams['use_cond_disc'] else None
        mel_p = model_out_with_gradient[way]['mel_out']
        if hasattr(self.model, 'out2mel'):
            mel_p = self.model.out2mel(mel_p)
        # Mel cheat
        o_ = self.mel_disc(mel_p, disc_cond_current_way)
        p_, pc_ = o_['y'], o_['y_c']
        if p_ is not None:
            log_outputs[f'{way}_a'] = self.mse_loss_fn(p_, p_.new_ones(p_.size()))
            loss_weights[f'{way}_a'] = hparams['lambda_mel_adv']
        if pc_ is not None:
            log_outputs[f'{way}_ac'] = self.mse_loss_fn(pc_, pc_.new_ones(pc_.size()))
            loss_weights[f'{way}_ac'] = hparams['lambda_mel_adv']
        # Spk cheat

        spk_emb_out = model_out_with_gradient[way]['h_style_out']
        o_ = self.spk_disc(mel_p, spk_emb_out)
        p_, pc_ = o_['y'], o_['y_c']
        if p_ is not None:
            log_outputs[f'{way}_spk'] = self.mse_loss_fn(p_, p_.new_ones(p_.size()))
            loss_weights[f'{way}_spk'] = hparams['lambda_mel_adv']
        if pc_ is not None:
            log_outputs[f'{way}_spkc'] = self.mse_loss_fn(pc_, pc_.new_ones(pc_.size()))
            loss_weights[f'{way}_spkc'] = hparams['lambda_mel_adv']

    def disc_judge_gen(self, way, sample, log_outputs):
        if hparams['rerun_gen']:
            exit(1)  # not allowed
        else:
            model_out = self.model_out_gt  # 这里是没有gradient的model_out.
        mel_g = self.get_corresponding_gtmel(way, sample)
        mel_p = model_out[way]['mel_out']
        if hasattr(self.model, 'out2mel'):
            mel_p = self.model.out2mel(mel_p)
        # Mel judge
        o = self.mel_disc(mel_g, self.disc_cond_gt[way])
        p, pc = o['y'], o['y_c']
        o_ = self.mel_disc(mel_p, self.disc_cond[way])
        p_, pc_ = o_['y'], o_['y_c']
        if p_ is not None:
            log_outputs[f"{way}_r"] = self.mse_loss_fn(p, p.new_ones(p.size()))
            log_outputs[f"{way}_f"] = self.mse_loss_fn(p_, p_.new_zeros(p_.size()))
        if pc_ is not None:
            log_outputs[f"{way}_rc"] = self.mse_loss_fn(pc, pc.new_ones(pc.size()))
            log_outputs[f"{way}_fc"] = self.mse_loss_fn(pc_, pc_.new_zeros(pc_.size()))
        # spk judge
        spk_emb_in = model_out[way]['h_style']
        spk_emb_out = model_out[way]['h_style_out']
        o_ = self.spk_disc(mel_g, spk_emb_in)
        p, pc = o['y'], o['y_c']
        o_ = self.spk_disc(mel_p, spk_emb_out)
        p_, pc_ = o_['y'], o_['y_c']
        if p_ is not None:
            log_outputs[f"{way}_spkr"] = self.mse_loss_fn(p, p.new_ones(p.size()))
            log_outputs[f"{way}_spkf"] = self.mse_loss_fn(p_, p_.new_zeros(p_.size()))
        if pc_ is not None:
            log_outputs[f"{way}_spkrc"] = self.mse_loss_fn(pc, pc.new_ones(pc.size()))
            log_outputs[f"{way}_spkfc"] = self.mse_loss_fn(pc_, pc_.new_zeros(pc_.size()))


class AmtSpkTask(ParaPPGPretrainedTask):
    def run_model(self, model, sample, tech_prefix=None, return_output=False):
        model.vc_asr.eval()
        #
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        spk_ids = sample['spk_ids'] if hparams['use_spk_id'] else None

        amateur_mels = sample['mels']  # [B, T_s, 80]
        amateur_pitch = sample['pitch']
        amateur_energy = sample['energy']
        amateur_tech_id = torch.zeros([amateur_mels.shape[0]], dtype=torch.long, device=amateur_mels.device)

        prof_mels = sample['prof_mels']
        prof_pitch = sample['prof_pitch']
        prof_energy = sample['prof_energy']
        prof_tech_id = torch.ones([prof_mels.shape[0]], dtype=torch.long, device=prof_mels.device)

        a2p_alignment = sample['a2p_f0_alignment']
        p2a_alignment = sample['p2a_f0_alignment']

        output = {}
        #  暂时不给energy.
        spk_ref_mel = amateur_mels
        if 'a2a' in self.concurrent_ways:
            a2a_output = self.model(mels_content=amateur_mels, mels_timbre=spk_ref_mel, pitch=amateur_pitch,
                                    energy=None, spk_ids=spk_ids, tech_ids=amateur_tech_id)
            output['a2a'] = a2a_output
        if 'p2p' in self.concurrent_ways:
            p2p_output = self.model(mels_content=prof_mels, mels_timbre=spk_ref_mel, pitch=prof_pitch,
                                    energy=None, spk_ids=spk_ids, tech_ids=prof_tech_id)
            output['p2p'] = p2p_output

        if 'a2p' in self.concurrent_ways:
            a2p_output = self.model(mels_content=amateur_mels, mels_timbre=spk_ref_mel, pitch=prof_pitch,
                                    energy=None, spk_ids=spk_ids, tech_ids=prof_tech_id,
                                    conversion_alignment=a2p_alignment)
            output['a2p'] = a2p_output

        if 'p2a' in self.concurrent_ways:
            p2a_output = self.model(mels_content=prof_mels, mels_timbre=spk_ref_mel, pitch=amateur_pitch,
                                    energy=None, spk_ids=spk_ids, tech_ids=amateur_tech_id,
                                    conversion_alignment=p2a_alignment)
            output['p2a'] = p2a_output

        losses = {}
        for way in self.concurrent_ways:
            mel_g = self.get_corresponding_gtmel(way, sample)
            self.add_mel_loss(output[way]['mel_out'], mel_g, losses, postfix=way)

        self.add_asr_losses(amateur_mels, prof_mels, txt_tokens, losses, a2p_alignment, p2a_alignment)
        if not return_output:
            return losses
        else:
            return losses, output


