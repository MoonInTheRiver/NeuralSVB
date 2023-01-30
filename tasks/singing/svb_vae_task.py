import numpy as np
import matplotlib
matplotlib.use('Agg')
import torch
from tasks.singing.neural_svb_task import FastSingingDataset
from vocoders.base_vocoder import get_vocoder_cls
import utils
from utils.hparams import hparams
from utils.pitch_utils import denorm_f0
from utils.common_schedulers import RSQRTSchedule, NoneSchedule
from modules.voice_conversion.svb_vae import SVBVAE, GlobalSVBVAE, MleSVBVAE
from tasks.singing.svb_para import ParaPPGPretrainedTask

import os
import json


class MultiSpkEmbDataset(FastSingingDataset):
    def __getitem__(self, index):
        sample = super(MultiSpkEmbDataset, self).__getitem__(index)
        item = self._get_item(index)
        a2p_f0_alignment = torch.LongTensor(item['a2p_f0_alignment'])[:sample['prof_pitch'].shape[0]].clip(
            max=sample['pitch'].shape[0] - 1)  # 最大值不超过amateur的长度 不然索引炸了.
        # p2a_f0_alignment = torch.LongTensor(item['p2a_f0_alignment'])[:sample['pitch'].shape[0]].clip(
        #     max=sample['prof_pitch'].shape[0] - 1)  # 最大值不超过prof的长度 不然索引炸了.
        assert a2p_f0_alignment.shape == sample['prof_pitch'].shape, (
        'a2p F0 alignment with unmatched shape: ', a2p_f0_alignment.shape, sample['prof_pitch'].shape)
        # assert p2a_f0_alignment.shape == sample['pitch'].shape, (
        # 'p2a F0 alignment with unmatched shape: ', p2a_f0_alignment.shape, sample['pitch'].shape)
        sample['a2p_f0_alignment'] = a2p_f0_alignment
        # sample['p2a_f0_alignment'] = p2a_f0_alignment
        # add spk emb
        multi_spk_emb = torch.FloatTensor(item['multi_spk_emb'])
        sample['multi_spk_emb'] = multi_spk_emb
        return sample

    def collater(self, samples):
        batch = super(MultiSpkEmbDataset, self).collater(samples)
        batch['a2p_f0_alignment'] = utils.collate_1d([s['a2p_f0_alignment'] for s in samples])
        # batch['p2a_f0_alignment'] = utils.collate_1d([s['p2a_f0_alignment'] for s in samples])
        # add spk emb
        batch['multi_spk_emb'] = utils.collate_2d([s['multi_spk_emb'] for s in samples])
        return batch


class SVBVAETask(ParaPPGPretrainedTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = MultiSpkEmbDataset

    def build_tts_model(self):
        data_dir = hparams['binary_data_dir']
        phone_list_file = os.path.join(data_dir, 'phone_set.json')
        phone_list = json.load(open(phone_list_file))
        self.model = SVBVAE(len(phone_list) + 10)

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

        self.gen_params = [p for name, p in self.model.named_parameters() if
                           ('vc_asr' not in name)
                           and
                           ('m_mapping_function' not in name)
                           and
                           ('logs_mapping_function' not in name)]  # list(self.model.parameters())

        self.mapping_params = list(self.model.m_mapping_function.parameters()) \
                              + list(self.model.logs_mapping_function.parameters())

        return self.model

    def configure_optimizers(self):
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
        optimizer_map = torch.optim.AdamW(
            self.mapping_params,
            lr=hparams['map_lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])
        self.scheduler = self.build_scheduler({'gen': optimizer_gen, 'disc': optimizer_disc, 'map': optimizer_map})
        return [optimizer_gen, optimizer_disc, optimizer_map]

    def gen_scheduler(self, optimizer):
        if hparams['scheduler'] == 'rsqrt':
            return RSQRTSchedule(optimizer)
        else:
            return NoneSchedule(optimizer)

    def build_scheduler(self, optimizer):
        return {
            "gen": self.gen_scheduler(optimizer["gen"]),
            "disc": torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer["disc"],
                **hparams["discriminator_scheduler_params"]) if optimizer["disc"] is not None else None,
            "map": torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer["map"],
                **hparams["map_scheduler_params"]),
        }

    def run_model(self, model, sample, concurrent_ways, tech_prefix=None, return_output=False, infer=False,
                  disable_map=False):
        model.vc_asr.eval()
        #
        # txt_tokens = sample['txt_tokens']  # [B, T_t]
        # spk_ids = sample['spk_ids'] if hparams['use_spk_id'] else None
        import pdb
        pdb.set_trace()
        amateur_mels = sample['mels']  # [B, T_s, 80]
        amateur_pitch = sample['pitch']
        # amateur_energy = sample['energy']
        # amateur_tech_id = torch.zeros([amateur_mels.shape[0]], dtype=torch.long, device=amateur_mels.device)

        prof_mels = sample['prof_mels']
        prof_pitch = sample['prof_pitch']
        # prof_energy = sample['prof_energy']
        # prof_tech_id = torch.ones([prof_mels.shape[0]], dtype=torch.long, device=prof_mels.device)

        a2p_alignment = sample['a2p_f0_alignment']
        # p2a_alignment = sample['p2a_f0_alignment']

        if infer:
            rand_spkemb_idx = 0
        else:
            rand_spkemb_idx = np.random.randint(1, sample['multi_spk_emb'].shape[1])  # range from [0, 4]
        spk_ids = sample['multi_spk_emb'][:, rand_spkemb_idx, :]  # [B, H]
        output = self.model(amateur_mel=amateur_mels, prof_mel=prof_mels,
                            amateur_pitch=amateur_pitch, prof_pitch=prof_pitch,
                            amateur_spk_id=spk_ids, prof_spk_id=spk_ids,  # both use amateur spk id
                            a2p_alignment=a2p_alignment, p2a_alignment=None,
                            infer=False, concurrent_ways=concurrent_ways,
                            disable_map=disable_map)  # 这里infer选了false. a2a, p2p都不是真正的infer.

        losses = {}
        for way in concurrent_ways:
            mel_g = self.get_corresponding_gtmel(way, sample)
            if 'kl' in output[way].keys():
                losses[f'{way}_kl'] = output[way]['kl'] * hparams['lambda_kl']
            if way not in ['a2a', 'p2p'] and hparams['cross_way_no_recon_loss']:
                pass
            else:
                self.add_mel_loss(output[way]['mel_out'], mel_g, losses, postfix=way)

        # self.add_asr_losses(amateur_mels, prof_mels, txt_tokens, losses, a2p_alignment, p2a_alignment)
        if not return_output:
            return losses
        else:
            return losses, output

    def _training_step(self, sample, batch_idx, optimizer_idx):
        log_outputs = {}
        loss_weights = {}
        disc_start = hparams['mel_gan'] and self.global_step > hparams["disc_start_steps"] and \
                     hparams['lambda_mel_adv'] > 0
        phase_1 = self.global_step < hparams['phase_1_steps']
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            if phase_1:
                self.model.m_mapping_function.eval()
                self.model.logs_mapping_function.eval()

                concurrent_ways = ['a2a', 'p2p']
                log_outputs, model_out = self.run_model(self.model, sample, concurrent_ways=concurrent_ways,
                                                        return_output=True)
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
                    for way in concurrent_ways:
                        self.gen_cheat_disc(way, model_out, log_outputs, loss_weights)
            if len(log_outputs) == 0:
                return None
        elif optimizer_idx == 1:
            #######################
            #    Discriminator    #
            #######################
            if phase_1:

                self.model.m_mapping_function.eval()
                self.model.logs_mapping_function.eval()
                concurrent_ways = ['a2a', 'p2p']
                if disc_start and self.global_step % hparams['disc_interval'] == 0:
                    for way in concurrent_ways:
                        self.disc_judge_gen(way, sample, log_outputs)

            if len(log_outputs) == 0:
                return None
        elif optimizer_idx == 2:
            #######################
            #   Mapping Function    #
            #######################
            if not phase_1:
                self.model.eval()
                self.model.m_mapping_function.train()
                self.model.logs_mapping_function.train()
                way = 'a2p'
                log_outputs, model_out = self.run_model(self.model, sample, concurrent_ways=['a2a', 'p2p', way],
                                                        return_output=True)
                cross_out = model_out[way]
                # recon loss & dist kl loss are in log_outputs

                # disc loss
                if hparams['cross_way_no_disc_loss']:
                    pass
                else:
                    a2p_sample_recon = cross_out['a2p_sample_recon']
                    p_ = self.mel_disc(a2p_sample_recon, None)['y']
                    if p_ is not None:
                        log_outputs[f'{way}_a'] = self.mse_loss_fn(p_, p_.new_ones(p_.size()))
                        loss_weights[f'{way}_a'] = hparams['lambda_mel_adv']
            if len(log_outputs) == 0:
                return None
        for way in ['a2a', 'p2p', 'a2p']:
            if f'{way}_kl' in log_outputs:
                if torch.any(torch.isnan(log_outputs[f'{way}_kl'])):
                    log_outputs[f'{way}_kl'] = log_outputs[f'{way}_kl'].detach()
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in log_outputs.items()])
        log_outputs['bs'] = sample['mels'].shape[0]
        return total_loss, log_outputs

    def vis_mel_tb(self, sample, batch_idx, mel_out, f0, sampling_rate, way, additional_name=''):
        _wav_out = self.vocoder.spec2wav(mel_out[0].cpu(), f0=f0[0].cpu())
        self.logger.add_audio(f'{way}_{additional_name}wavout_{batch_idx}', _wav_out, self.global_step, sampling_rate)
        mel_g, tgtprefix = self.get_corresponding_gtmel(way, sample, return_tgtprefix=True)
        self.plot_mel(batch_idx, mel_g, mel_out, name=f'{way}_gt_{batch_idx}')

        ### here for debugging
        self.pitch_debugging(mel_out, mel_g, sample[f'{tgtprefix}f0'], sample[f'{tgtprefix}uv'],
                             name=f'{way}_{additional_name}f0_w_mel_{batch_idx}')

    def validation_step(self, sample, batch_idx):
        prof_mel = sample['prof_mels']
        prof_f0s = denorm_f0(sample['prof_f0'], sample['prof_uv'], hparams)

        amateur_mel = sample['mels']
        amateur_f0s = denorm_f0(sample['f0'], sample['uv'], hparams)

        f0s = {
            'a2a': amateur_f0s,
            'p2p': prof_f0s,
            'a2p': prof_f0s
        }
        # vmin = hparams['mel_vmin']
        # vmax = hparams['mel_vmax']
        outputs = {}
        outputs['losses'] = {}
        self.watch_asr_loss = True

        concurrent_ways = ['a2a', 'p2p', 'a2p']
        outputs['losses'], model_out = self.run_model(self.model, sample, concurrent_ways=concurrent_ways,
                                                      return_output=True, infer=True,
                                                      disable_map=hparams['disable_map'])
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.tensors_to_scalars(outputs)
        if self.global_step % hparams['valid_infer_interval'] == 0 \
                and batch_idx < hparams['num_valid_plots']:
            if self.vocoder is None:
                self.vocoder = get_vocoder_cls(hparams)()
            sampling_rate = hparams['audio_sample_rate']

            for way in concurrent_ways:
                self.vis_mel_tb(sample, batch_idx, model_out[way]['mel_out'], f0s[way], sampling_rate, way,
                                additional_name='')
                if way not in ['a2a', 'p2p']:
                    self.vis_mel_tb(sample, batch_idx, model_out[way]['a2p_sample_recon'], f0s[way], sampling_rate, way,
                                    additional_name='sampled_')

            gt_a_wav_out = self.vocoder.spec2wav(amateur_mel[0].cpu(), f0=amateur_f0s[0].cpu())
            gt_p_wav_out = self.vocoder.spec2wav(prof_mel[0].cpu(), f0=prof_f0s[0].cpu())
            self.logger.add_audio(f'gt_a_wav_{batch_idx}', gt_a_wav_out, self.global_step, sampling_rate)
            self.logger.add_audio(f'gt_p_wav_{batch_idx}', gt_p_wav_out, self.global_step, sampling_rate)

        return outputs

    def test_step(self, sample, batch_idx):
        concurrent_ways = ['a2a', 'p2p', 'a2p']
        _, model_out = self.run_model(self.model, sample, concurrent_ways=concurrent_ways, return_output=True,
                                      infer=True, disable_map=hparams['disable_map'])

        prof_f0s = denorm_f0(sample['prof_f0'], sample['prof_uv'], hparams)
        amateur_f0s = denorm_f0(sample['f0'], sample['uv'], hparams)
        f0s = {
            'a2a_f0s': amateur_f0s,
            'p2p_f0s': prof_f0s,
            'a2p_f0s': prof_f0s,
            'p2a_f0s': amateur_f0s
        }
        mels = {}
        for way in concurrent_ways:
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
        mels_dict = {
            'gt_a_mel': prediction['mels'],
            'gt_p_mel': prediction['prof_mels'],
        }
        concurrent_ways = ['a2a', 'p2p', 'a2p']

        for way in concurrent_ways:
            wavs_dict[f'{way}_wavout'] = self.vocoder.spec2wav(prediction[f'{way}_mels'], f0=prediction[f'{way}_f0s'])
            mels_dict[f'{way}_mel'] = prediction[f'{way}_mels']

        os.makedirs(gen_dir, exist_ok=True)
        os.makedirs(f'{gen_dir}/mels', exist_ok=True)
        os.makedirs(f'{gen_dir}/wavs', exist_ok=True)
        for key in wavs_dict:
            if hparams['disable_map']:
                os.makedirs(f'{gen_dir}/wavs/disable_map_{key}', exist_ok=True)
            else:
                os.makedirs(f'{gen_dir}/wavs/{key}', exist_ok=True)
        for key in mels_dict:
            if hparams['disable_map']:
                os.makedirs(f'{gen_dir}/mels/disable_map_{key}', exist_ok=True)
            else:
                os.makedirs(f'{gen_dir}/mels/{key}', exist_ok=True)

        self.saving_results_futures.append(
            self.saving_result_pool.apply_async(self.save_result, args=[
                wavs_dict, base_fn % 'P', gen_dir, str_phs, mels_dict]))

        self.results_id += 1
        return {
            'item_name': item_name,
            'text': text,
        }


class SVBVAEBoostTask(SVBVAETask):

    def __init__(self):
        super().__init__()
        self.dataset_cls = MultiSpkEmbDataset

    def on_before_optimization(self, opt_idx):
        if opt_idx == 0:
            torch.nn.utils.clip_grad_norm_(self.gen_params, hparams['generator_grad_norm'])
        elif opt_idx == 1:
            torch.nn.utils.clip_grad_norm_(self.disc_params, hparams["discriminator_grad_norm"])
        elif opt_idx == 2:
            torch.nn.utils.clip_grad_norm_(self.mapping_params, hparams['generator_grad_norm'])

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if optimizer_idx == 0:
            self.scheduler['gen'].step(self.global_step)
        elif optimizer_idx == 1:
            self.scheduler['disc'].step(max(self.global_step - hparams["disc_start_steps"], 1))
        elif optimizer_idx == 2:
            self.scheduler['map'].step(self.global_step)

    def build_tts_model(self):
        data_dir = hparams['binary_data_dir']
        phone_list_file = os.path.join(data_dir, 'phone_set.json')
        phone_list = json.load(open(phone_list_file))
        self.model = GlobalSVBVAE(len(phone_list) + 10)

    def _training_step(self, sample, batch_idx, optimizer_idx):
        log_outputs = {}
        loss_weights = {}
        disc_start = hparams['mel_gan'] and self.global_step > hparams["disc_start_steps"] and \
                     hparams['lambda_mel_adv'] > 0
        phase_1, phase_2, phase_3 = False, False, False
        concurrent_ways = []
        if self.global_step <= hparams['phase_1_steps']:
            phase_1 = True
            concurrent_ways = hparams['phase_1_concurrent_ways'].split(',')
        elif hparams['phase_1_steps'] < self.global_step <= hparams['phase_2_steps']:
            phase_2 = True
            concurrent_ways = hparams['phase_2_concurrent_ways'].split(',')
        elif hparams['phase_2_steps'] < self.global_step:
            phase_3 = True
            concurrent_ways = hparams['phase_3_concurrent_ways'].split(',')
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            if phase_1 or phase_2:
                self.model.m_mapping_function.eval()
                self.model.logs_mapping_function.eval()
                log_outputs, model_out = self.run_model(self.model, sample, concurrent_ways=concurrent_ways,
                                                        return_output=True)
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
                    for way in concurrent_ways:
                        self.gen_cheat_disc(way, model_out, log_outputs, loss_weights)
            if len(log_outputs) == 0:
                return None
        elif optimizer_idx == 1:
            #######################
            #    Discriminator    #
            #######################
            if phase_1 or phase_2:
                self.model.m_mapping_function.eval()
                self.model.logs_mapping_function.eval()
                if disc_start and self.global_step % hparams['disc_interval'] == 0:
                    for way in concurrent_ways:
                        self.disc_judge_gen(way, sample, log_outputs)
            if len(log_outputs) == 0:
                return None
        elif optimizer_idx == 2:
            #######################
            #   Mapping Function    #
            #######################
            if phase_3:
                self.model.eval()
                self.model.m_mapping_function.train()
                self.model.logs_mapping_function.train()
                log_outputs, model_out = self.run_model(self.model, sample,
                                                        concurrent_ways=['a2a', 'p2p'] + concurrent_ways,
                                                        return_output=True)
                for way in concurrent_ways:
                    cross_out = model_out[way]
                    # recon loss & dist kl loss are in log_outputs
                    # disc loss
                    if hparams['cross_way_no_disc_loss']:
                        pass
                    else:
                        cross_sample_recon = cross_out[f'{way}_sample_recon']
                        p_ = self.mel_disc(cross_sample_recon, None)['y']
                        if p_ is not None:
                            log_outputs[f'{way}_a'] = self.mse_loss_fn(p_, p_.new_ones(p_.size()))
                            loss_weights[f'{way}_a'] = hparams['lambda_mel_adv']
            if len(log_outputs) == 0:
                return None

        for way in ['a2a', 'p2p', 'a2p']:
            if f'{way}_kl' in log_outputs:
                if torch.any(torch.isnan(log_outputs[f'{way}_kl'])):
                    log_outputs[f'{way}_kl'] = log_outputs[f'{way}_kl'].detach()

        total_loss = sum([loss_weights.get(k, 1) * v for k, v in log_outputs.items()])
        log_outputs['bs'] = sample['mels'].shape[0]
        return total_loss, log_outputs

    def validation_step(self, sample, batch_idx):
        prof_mel = sample['prof_mels']
        prof_f0s = denorm_f0(sample['prof_f0'], sample['prof_uv'], hparams)
        amateur_mel = sample['mels']
        amateur_f0s = denorm_f0(sample['f0'], sample['uv'], hparams)
        f0s = {
            'a2a': amateur_f0s,
            'p2p': prof_f0s,
            'a2p': prof_f0s
        }
        outputs = {}
        outputs['losses'] = {}
        self.watch_asr_loss = True
        concurrent_ways = []
        if self.global_step <= hparams['phase_1_steps']:
            concurrent_ways = ['p2p']
        elif hparams['phase_1_steps'] < self.global_step <= hparams['phase_2_steps']:
            concurrent_ways = ['a2a', 'p2p', 'a2p']
        elif hparams['phase_2_steps'] < self.global_step:
            concurrent_ways = ['a2a', 'p2p', 'a2p']
        outputs['losses'], model_out = self.run_model(self.model, sample, concurrent_ways=concurrent_ways,
                                                      return_output=True, infer=True,
                                                      disable_map=hparams['disable_map'])
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.tensors_to_scalars(outputs)
        if self.global_step % hparams['valid_infer_interval'] == 0 \
                and batch_idx < hparams['num_valid_plots']:
            if self.vocoder is None:
                self.vocoder = get_vocoder_cls(hparams)()
            sampling_rate = hparams['audio_sample_rate']
            for way in concurrent_ways:
                self.vis_mel_tb(sample, batch_idx, model_out[way]['mel_out'], f0s[way], sampling_rate, way,
                                additional_name='')
                if 'a2p_sample_recon' in model_out[way].keys():
                    self.vis_mel_tb(sample, batch_idx, model_out[way]['a2p_sample_recon'], f0s[way], sampling_rate, way,
                                    additional_name='sampled_')
            gt_a_wav_out = self.vocoder.spec2wav(amateur_mel[0].cpu(), f0=amateur_f0s[0].cpu())
            gt_p_wav_out = self.vocoder.spec2wav(prof_mel[0].cpu(), f0=prof_f0s[0].cpu())
            self.logger.add_audio(f'gt_a_wav_{batch_idx}', gt_a_wav_out, self.global_step, sampling_rate)
            self.logger.add_audio(f'gt_p_wav_{batch_idx}', gt_p_wav_out, self.global_step, sampling_rate)

        return outputs


class SVBVAEMleTask(SVBVAEBoostTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = MultiSpkEmbDataset

    def build_tts_model(self):
        data_dir = hparams['binary_data_dir']
        phone_list_file = os.path.join(data_dir, 'phone_set.json')
        phone_list = json.load(open(phone_list_file))
        self.model = GlobalSVBVAE(len(phone_list) + 10)
        self.model = MleSVBVAE(len(phone_list) + 10)

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

        self.gen_params = [p for name, p in self.model.named_parameters() if
                           ('vc_asr' not in name)
                           and
                           ('z_mapping_function' not in name)
                           ]  # list(self.model.parameters())

        self.mapping_params = list(self.model.z_mapping_function.parameters())

        return self.model

    def _training_step(self, sample, batch_idx, optimizer_idx):
        log_outputs = {}
        loss_weights = {}
        disc_start = hparams['mel_gan'] and self.global_step > hparams["disc_start_steps"] and \
                     hparams['lambda_mel_adv'] > 0

        phase_1, phase_2, phase_3 = False, False, False
        concurrent_ways = []
        if self.global_step <= hparams['phase_1_steps']:
            phase_1 = True
            concurrent_ways = hparams['phase_1_concurrent_ways'].split(',')
        elif hparams['phase_1_steps'] < self.global_step <= hparams['phase_2_steps']:
            phase_2 = True
            concurrent_ways = hparams['phase_2_concurrent_ways'].split(',')
        elif hparams['phase_2_steps'] < self.global_step:
            phase_3 = True
            concurrent_ways = hparams['phase_3_concurrent_ways'].split(',')

        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            if phase_1 or phase_2:
                self.model.z_mapping_function.eval()

                log_outputs, model_out = self.run_model(self.model, sample, concurrent_ways=concurrent_ways,
                                                        return_output=True)
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
                    for way in concurrent_ways:
                        self.gen_cheat_disc(way, model_out, log_outputs, loss_weights)
            if len(log_outputs) == 0:
                return None
        elif optimizer_idx == 1:
            #######################
            #    Discriminator    #
            #######################
            if phase_1 or phase_2:
                self.model.z_mapping_function.eval()

                if disc_start and self.global_step % hparams['disc_interval'] == 0:
                    for way in concurrent_ways:
                        self.disc_judge_gen(way, sample, log_outputs)

            if len(log_outputs) == 0:
                return None
        elif optimizer_idx == 2:
            #######################
            #   Mapping Function    #
            #######################
            if phase_3:
                self.model.eval()
                self.model.z_mapping_function.train()

                log_outputs, model_out = self.run_model(self.model, sample,
                                                        concurrent_ways=['a2a', 'p2p'] + concurrent_ways,
                                                        return_output=True)
                for way in concurrent_ways:
                    cross_out = model_out[way]
                    # recon loss are in log_outputs

                    # mle loss
                    log_outputs[f'{way}_mle'] = cross_out['mle']
                    loss_weights[f'{way}_mle'] = hparams['lambda_mle']

                    # disc loss
                    if hparams['cross_way_no_disc_loss']:
                        pass
                    else:
                        cross_recon = cross_out['mel_out']
                        p_ = self.mel_disc(cross_recon, None)['y']
                        if p_ is not None:
                            log_outputs[f'{way}_a'] = self.mse_loss_fn(p_, p_.new_ones(p_.size()))
                            loss_weights[f'{way}_a'] = hparams['lambda_mel_adv']
            if len(log_outputs) == 0:
                return None

        for way in ['a2a', 'p2p', 'a2p']:
            if f'{way}_kl' in log_outputs:
                if torch.any(torch.isnan(log_outputs[f'{way}_kl'])):
                    log_outputs[f'{way}_kl'] = log_outputs[f'{way}_kl'].detach()

        total_loss = sum([loss_weights.get(k, 1) * v for k, v in log_outputs.items()])
        log_outputs['bs'] = sample['mels'].shape[0]
        return total_loss, log_outputs

    def validation_step(self, sample, batch_idx):
        prof_mel = sample['prof_mels']
        prof_f0s = denorm_f0(sample['prof_f0'], sample['prof_uv'], hparams)

        amateur_mel = sample['mels']
        amateur_f0s = denorm_f0(sample['f0'], sample['uv'], hparams)

        f0s = {
            'a2a': amateur_f0s,
            'p2p': prof_f0s,
            'a2p': prof_f0s
        }
        outputs = {}
        outputs['losses'] = {}
        self.watch_asr_loss = True

        concurrent_ways = []
        if self.global_step <= hparams['phase_1_steps']:
            concurrent_ways = ['p2p']
        elif hparams['phase_1_steps'] < self.global_step <= hparams['phase_2_steps']:
            concurrent_ways = ['a2a', 'p2p', 'a2p']
        elif hparams['phase_2_steps'] < self.global_step:
            concurrent_ways = ['a2a', 'p2p', 'a2p']

        outputs['losses'], model_out = self.run_model(self.model, sample, concurrent_ways=concurrent_ways,
                                                      return_output=True, infer=True,
                                                      disable_map=hparams['disable_map'])
        for way in concurrent_ways:
            if 'mle' in model_out[way]:
                outputs['losses'][f'{way}_mle'] = model_out[way]['mle']
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.tensors_to_scalars(outputs)
        if self.global_step % hparams['valid_infer_interval'] == 0 \
                and batch_idx < hparams['num_valid_plots']:
            if self.vocoder is None:
                self.vocoder = get_vocoder_cls(hparams)()
            sampling_rate = hparams['audio_sample_rate']

            for way in concurrent_ways:
                self.vis_mel_tb(sample, batch_idx, model_out[way]['mel_out'], f0s[way], sampling_rate, way,
                                additional_name='')

            gt_a_wav_out = self.vocoder.spec2wav(amateur_mel[0].cpu(), f0=amateur_f0s[0].cpu())
            gt_p_wav_out = self.vocoder.spec2wav(prof_mel[0].cpu(), f0=prof_f0s[0].cpu())
            self.logger.add_audio(f'gt_a_wav_{batch_idx}', gt_a_wav_out, self.global_step, sampling_rate)
            self.logger.add_audio(f'gt_p_wav_{batch_idx}', gt_p_wav_out, self.global_step, sampling_rate)

        return outputs



